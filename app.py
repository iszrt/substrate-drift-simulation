import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import time
import datetime
from collections import defaultdict
from tsd import ETFCSA_TSD
from validate_schedule import (
    load_schedule, validate, format_report,
    classify_room, LAB_SUBJECTS, parse_time_range,
)

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="SDCSA Scheduler", layout="wide")

st.title("University Class Scheduler")
st.markdown("**Powered by Substrate Drift Clonal Selection Algorithm**")

# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("Algorithm Parameters")
st.sidebar.markdown("Adjust hyperparameters for your thesis algorithm.")

N = st.sidebar.slider("Population Size (N)", min_value=10, max_value=300, value=200, step=10)
max_evals = st.sidebar.number_input("Max Evaluations", min_value=1000, max_value=500000, value=150000, step=10000)
n_clones = st.sidebar.slider("Clone Rate (n_clones)", min_value=1, max_value=20, value=10)
rho = st.sidebar.slider("Substrate Decay (rho)", min_value=0.5, max_value=0.99, value=0.98, step=0.01)

# Soft constraint weights
st.sidebar.header("Constraint Weights")
st.sidebar.markdown("Weights for soft constraints (0 = disabled).")
w_room_type = st.sidebar.slider("Room Type Mismatch (S1)", 0.0, 5.0, 0.5, 0.1)
w_daily_load = st.sidebar.slider("Section Daily Overload (S2)", 0.0, 5.0, 0.3, 0.1)
w_inst_load = st.sidebar.slider("Instructor Daily Overload (S3)", 0.0, 5.0, 0.3, 0.1)
w_gap = st.sidebar.slider("Schedule Gap > 2h (S4)", 0.0, 5.0, 0.2, 0.1)
w_late = st.sidebar.slider("Late Class >= 6PM (S5)", 0.0, 5.0, 0.1, 0.1)

# --- MAIN UI ---
st.header("1. Upload Class Requirements Data")
st.markdown("Upload the **cleaned** input file (`Algorithm_Input_Cleaned.xlsx`) from the `cleaned/` folder.")
uploaded_file = st.file_uploader("Upload Algorithm Input Excel", type=["xlsx"])

if uploaded_file is not None:
    df_classes = pd.read_excel(uploaded_file, sheet_name='Class_Requirements')
    df_rooms = pd.read_excel(uploaded_file, sheet_name='Room_List')

    # Detect Mode column
    has_mode = 'Mode' in df_classes.columns
    if has_mode:
        n_f2f = (df_classes['Mode'] == 'face-to-face').sum()
        n_online = (df_classes['Mode'] == 'online').sum()
        st.success(
            f"Data loaded: {len(df_classes)} class meetings "
            f"({n_f2f} face-to-face, {n_online} online), "
            f"{len(df_rooms)} rooms."
        )
    else:
        st.success(f"Data loaded: {len(df_classes)} classes, {len(df_rooms)} rooms.")
        st.warning("No 'Mode' column found. All classes will be treated as face-to-face. "
                    "Use the cleaned input from `preprocess.py` for proper online class handling.")

    if st.button("Run Optimization"):

        classes = df_classes.to_dict('records')
        rooms = df_rooms['Room'].tolist()
        num_classes = len(classes)
        num_rooms = len(rooms)

        # --- Build room type lookup for soft constraint S1 ---
        room_type_map = {r: classify_room(r) for r in rooms}
        lab_room_indices = [i for i, r in enumerate(rooms) if room_type_map[r] == 'lab']

        # --- Identify online vs face-to-face classes ---
        is_online_class = []
        for c in classes:
            if has_mode:
                is_online_class.append(str(c.get('Mode', '')).strip().lower() == 'online')
            else:
                is_online_class.append(False)

        num_f2f = sum(1 for x in is_online_class if not x)
        num_online = sum(1 for x in is_online_class if x)

        # --- Timeslot grid ---
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        times = [
            "07:00 AM - 08:30 AM", "08:30 AM - 10:00 AM", "10:00 AM - 11:30 AM",
            "11:30 AM - 01:00 PM", "01:00 PM - 02:30 PM", "02:30 PM - 04:00 PM",
            "04:00 PM - 05:30 PM", "05:30 PM - 07:00 PM", "07:00 PM - 08:30 PM",
            "08:30 PM - 10:00 PM"
        ]
        timeslots = [{'Day': d, 'Time': t} for d in days for t in times]
        num_timeslots = len(timeslots)

        # Pre-parse timeslot start minutes for soft constraints
        timeslot_start_min = []
        for ts in timeslots:
            tr = parse_time_range(ts['Time'])
            timeslot_start_min.append(tr[0] if tr else 0)

        # Day index for each timeslot (0=Mon..5=Sat)
        day_to_idx = {d: i for i, d in enumerate(days)}
        timeslot_day_idx = [day_to_idx[ts['Day']] for ts in timeslots]

        # --- Encoding ---
        # Face-to-face: 2 vars (timeslot + room)
        # Online: 1 var (timeslot only, no room needed)
        dim = num_f2f * 2 + num_online * 1

        # Build index mapping: for class i, where are its variables in x?
        var_offsets = []  # (ts_offset, rm_offset_or_None)
        idx = 0
        for i in range(num_classes):
            if is_online_class[i]:
                var_offsets.append((idx, None))
                idx += 1
            else:
                var_offsets.append((idx, idx + 1))
                idx += 2

        def decode_schedule(x: np.ndarray):
            schedule = []
            for i in range(num_classes):
                ts_off, rm_off = var_offsets[i]
                ts_idx = int(np.clip(x[ts_off] * num_timeslots, 0, num_timeslots - 1))

                if rm_off is not None:
                    rm_idx = int(np.clip(x[rm_off] * num_rooms, 0, num_rooms - 1))
                    room_name = rooms[rm_idx]
                else:
                    rm_idx = -1  # online, no room
                    room_name = "ONLINE"

                schedule.append({
                    'Department': classes[i].get('Department', ''),
                    'Section': classes[i]['Section'],
                    'Subject': classes[i]['Subject'],
                    'Instructor': classes[i]['Instructor'],
                    'Day': timeslots[ts_idx]['Day'],
                    'Time': timeslots[ts_idx]['Time'],
                    'Room': room_name,
                    'Timeslot_Idx': ts_idx,
                    'Room_Idx': rm_idx,
                    'Is_Online': is_online_class[i],
                })
            return schedule

        def calculate_conflicts(x: np.ndarray) -> float:
            schedule = decode_schedule(x)
            hard_penalty = 0
            soft_penalty = 0.0

            room_usage = set()
            instructor_usage = set()
            section_usage = set()

            # Per-day tracking for soft constraints
            section_day_count = {}     # (section, day_idx) -> count
            instructor_day_count = {}  # (instructor, day_idx) -> count
            section_day_slots = {}     # (section, day_idx) -> [start_minutes...]

            for entry in schedule:
                ts = entry['Timeslot_Idx']
                rm = entry['Room_Idx']
                inst = entry['Instructor']
                sec = entry['Section']
                subj = str(entry.get('Subject', ''))
                online = entry['Is_Online']
                d_idx = timeslot_day_idx[ts]
                start_min = timeslot_start_min[ts]

                # --- H1: Room Conflict (skip online) ---
                if not online:
                    if (rm, ts) in room_usage:
                        hard_penalty += 1
                    else:
                        room_usage.add((rm, ts))

                # --- H2: Instructor Conflict ---
                if str(inst).upper() not in ['NAN', 'NONE', 'NO TEACHER', 'UNKNOWN']:
                    if (inst, ts) in instructor_usage:
                        hard_penalty += 1
                    else:
                        instructor_usage.add((inst, ts))

                # --- H3: Section Conflict ---
                if (sec, ts) in section_usage:
                    hard_penalty += 1
                else:
                    section_usage.add((sec, ts))

                # --- S1: Room Type Mismatch (skip online) ---
                if w_room_type > 0 and not online:
                    if subj in LAB_SUBJECTS:
                        if room_type_map.get(rooms[rm], 'other') != 'lab':
                            soft_penalty += w_room_type

                # --- S5: Late Class ---
                if w_late > 0 and start_min >= 1080:  # 6:00 PM
                    soft_penalty += w_late

                # Track daily loads
                key_sec = (sec, d_idx)
                section_day_count[key_sec] = section_day_count.get(key_sec, 0) + 1

                if key_sec not in section_day_slots:
                    section_day_slots[key_sec] = []
                section_day_slots[key_sec].append(start_min)

                if str(inst).upper() not in ['NAN', 'NONE', 'NO TEACHER', 'UNKNOWN']:
                    key_inst = (inst, d_idx)
                    instructor_day_count[key_inst] = instructor_day_count.get(key_inst, 0) + 1

            # --- S2: Section Daily Overload ---
            if w_daily_load > 0:
                for count in section_day_count.values():
                    if count > 5:
                        soft_penalty += w_daily_load * (count - 5)

            # --- S3: Instructor Daily Overload ---
            if w_inst_load > 0:
                for count in instructor_day_count.values():
                    if count > 5:
                        soft_penalty += w_inst_load * (count - 5)

            # --- S4: Schedule Gap ---
            if w_gap > 0:
                for slots in section_day_slots.values():
                    if len(slots) < 2:
                        continue
                    sorted_slots = sorted(slots)
                    for k in range(len(sorted_slots) - 1):
                        # Each timeslot is 90 min, so the end is start + 90
                        gap = sorted_slots[k + 1] - (sorted_slots[k] + 90)
                        if gap > 120:  # > 2 hours
                            soft_penalty += w_gap

            return float(hard_penalty) + soft_penalty

        st.header("2. Optimizing Schedule")
        st.info(f"Problem size: {num_classes} meetings ({num_f2f} face-to-face + {num_online} online), "
                f"{dim} decision variables, {num_rooms} rooms, {num_timeslots} timeslots")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(**kwargs):
            current_evals = kwargs.get('evals', 0)
            progress = min(current_evals / max_evals, 1.0)
            progress_bar.progress(progress)
            status_text.markdown(f"**Evaluations:** {current_evals} / {max_evals}...")

        bounds = [(0.0, 1.0) for _ in range(dim)]
        optimizer = ETFCSA_TSD(
            func=calculate_conflicts,
            bounds=bounds,
            N=N,
            max_evals=max_evals,
            n_clones=n_clones,
            rho=rho,
            c_threshold=3.0,
            eta=0.25,
            progress=update_progress
        )

        start_time = time.time()
        best_x, best_f, info = optimizer.optimize()
        runtime = time.time() - start_time

        progress_bar.progress(1.0)
        status_text.success(f"Optimization Complete! Final Fitness: {best_f:.2f} | Time: {runtime:.2f}s")

        best_schedule = decode_schedule(best_x)

        # --- LOCAL VALIDATION (using validate_schedule.py) ---
        st.header("3. Schedule Validation")
        st.markdown("Independent validation using the same framework applied to the initial CITC schedule.")

        # Build a DataFrame in the same format as CITC_SCHEDULING_CLEANED
        val_records = []
        for entry in best_schedule:
            val_records.append({
                'Day': entry['Day'],
                'Time': entry['Time'],
                'Subject': entry['Subject'],
                'Room': entry['Room'] if entry['Room'] != 'ONLINE' else None,
                'Instructor': entry['Instructor'],
                'Section': entry['Section'],
            })
        df_val = pd.DataFrame(val_records)

        report = validate(df_val, schedule_name="Optimized Schedule")

        # --- Results Overview ---
        st.subheader("Results Overview")

        # Feasibility banner
        if report.is_feasible:
            st.success("FEASIBLE -- All hard constraints satisfied!")
        else:
            st.error(f"INFEASIBLE -- {report.hard_count} hard constraint violations remain.")

        # Summary metrics row
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Final Fitness", f"{best_f:.2f}")
        with r2:
            st.metric("Hard Violations", report.hard_count)
        with r3:
            st.metric("Soft Violations", report.soft_penalty)
        with r4:
            st.metric("Runtime", f"{runtime:.1f}s")

        # --- Comparison with initial schedule ---
        st.subheader("Before vs After Optimization")
        cmp1, cmp2, cmp3 = st.columns(3)
        with cmp1:
            st.metric("Hard Violations", report.hard_count, delta=f"{report.hard_count - 380}", delta_color="inverse")
            st.caption("Initial: 380")
        with cmp2:
            st.metric("Soft Violations", report.soft_penalty, delta=f"{report.soft_penalty - 245}", delta_color="inverse")
            st.caption("Initial: 245")
        with cmp3:
            feas_now = "FEASIBLE" if report.is_feasible else "INFEASIBLE"
            st.metric("Feasibility", feas_now)
            st.caption("Initial: INFEASIBLE")

        # --- Hard Constraint Breakdown ---
        st.subheader("Hard Constraint Breakdown")
        hard_groups = defaultdict(list)
        for v in report.hard_violations:
            hard_groups[v.constraint].append(v)

        h1, h2, h3 = st.columns(3)
        hard_labels = [
            ("H1: Room Conflict", h1),
            ("H2: Instructor Conflict", h2),
            ("H3: Section Conflict", h3),
        ]
        for cname, col in hard_labels:
            vlist = hard_groups.get(cname, [])
            with col:
                if len(vlist) == 0:
                    st.success(f"{cname}\n\n**0** violations")
                else:
                    st.error(f"{cname}\n\n**{len(vlist)}** violations")
                with st.expander(f"View details"):
                    if len(vlist) == 0:
                        st.write("No violations.")
                    else:
                        for v in vlist[:30]:
                            st.text(v.description)
                        if len(vlist) > 30:
                            st.text(f"... and {len(vlist) - 30} more")

        # --- Soft Constraint Breakdown ---
        st.subheader("Soft Constraint Breakdown")
        st.markdown(f"**Active weights:** S1={w_room_type}, S2={w_daily_load}, S3={w_inst_load}, S4={w_gap}, S5={w_late}")

        soft_groups = defaultdict(list)
        for v in report.soft_violations:
            soft_groups[v.constraint].append(v)

        s1, s2, s3, s4, s5 = st.columns(5)
        soft_labels = [
            ("S1: Room Type Mismatch", s1, w_room_type),
            ("S2: Section Daily Overload", s2, w_daily_load),
            ("S3: Instructor Daily Overload", s3, w_inst_load),
            ("S4: Schedule Gap", s4, w_gap),
            ("S5: Late Class", s5, w_late),
        ]
        for cname, col, weight in soft_labels:
            vlist = soft_groups.get(cname, [])
            with col:
                if weight == 0:
                    st.warning(f"{cname.split(': ')[1]}\n\n**DISABLED**")
                elif len(vlist) == 0:
                    st.success(f"{cname.split(': ')[1]}\n\n**0**")
                else:
                    st.info(f"{cname.split(': ')[1]}\n\n**{len(vlist)}**")
                with st.expander("Details"):
                    if weight == 0:
                        st.write("Weight set to 0 (disabled).")
                    elif len(vlist) == 0:
                        st.write("No violations.")
                    else:
                        for v in vlist[:20]:
                            st.text(v.description)
                        if len(vlist) > 20:
                            st.text(f"... and {len(vlist) - 20} more")

        # Warnings
        if report.warnings:
            st.subheader("Warnings")
            for w in report.warnings:
                st.warning(w)

        # --- Algorithm Parameters Used ---
        with st.expander("Algorithm Parameters Used"):
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Population (N)", N)
            with p2:
                st.metric("Max Evaluations", f"{max_evals:,}")
            with p3:
                st.metric("Clones", n_clones)
            with p4:
                st.metric("Rho (decay)", rho)

        # --- EXPORT OPTIONS ---
        st.header("4. Download Output")

        col1 = st.columns(1)[0]

        df_best = pd.DataFrame(best_schedule)
        day_map_excel = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
        df_best['Day_Num'] = df_best['Day'].map(day_map_excel)
        export_cols = ['Department', 'Section', 'Subject', 'Instructor', 'Day', 'Time', 'Room']
        df_final = df_best.sort_values(by=['Department', 'Section', 'Day_Num', 'Time'])[export_cols]

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_final.to_excel(writer, sheet_name='ALL_CLASSES', index=False)
            for sec in sorted(df_final['Section'].unique()):
                df_sec = df_final[df_final['Section'] == sec]
                safe_name = str(sec)[:31]
                df_sec.to_excel(writer, sheet_name=safe_name, index=False)

        with col1:
            st.download_button(
                label="Download Excel Format",
                data=output.getvalue(),
                file_name="Optimized_Schedule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # Full validation report download
        report_text = format_report(report)
        st.download_button(
            label="Download Full Validation Report",
            data=report_text,
            file_name="validation_report.txt",
            mime="text/plain",
        )

        # --- SAVE LOG FILE ---
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"run_{timestamp}.log")

        hard_groups_log = defaultdict(int)
        for v in report.hard_violations:
            hard_groups_log[v.constraint] += 1
        soft_groups_log = defaultdict(int)
        for v in report.soft_violations:
            soft_groups_log[v.constraint] += 1

        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"=== ETFCSA-TSD Optimization Log ===\n")
            lf.write(f"Timestamp   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            lf.write(f"Runtime     : {runtime:.2f}s\n")
            lf.write(f"Final Fitness: {best_f:.2f}\n\n")
            lf.write(f"--- Parameters ---\n")
            lf.write(f"Population  : {N}\n")
            lf.write(f"Max Evals   : {max_evals}\n")
            lf.write(f"Clones      : {n_clones}\n")
            lf.write(f"Rho         : {rho}\n")
            lf.write(f"Weights     : S1={w_room_type}, S2={w_daily_load}, S3={w_inst_load}, S4={w_gap}, S5={w_late}\n\n")
            lf.write(f"--- Problem Size ---\n")
            lf.write(f"Total Classes: {num_classes} ({num_f2f} f2f + {num_online} online)\n")
            lf.write(f"Variables    : {dim}\n")
            lf.write(f"Rooms        : {num_rooms}\n")
            lf.write(f"Timeslots    : {num_timeslots}\n\n")
            lf.write(f"--- Validation Results ---\n")
            feasible_str = "FEASIBLE" if report.is_feasible else "INFEASIBLE"
            lf.write(f"Feasibility  : {feasible_str}\n")
            lf.write(f"Hard Violations: {report.hard_count}\n")
            for cname in ["H1: Room Conflict", "H2: Instructor Conflict", "H3: Section Conflict"]:
                lf.write(f"  {cname}: {hard_groups_log.get(cname, 0)}\n")
            lf.write(f"Soft Violations: {report.soft_penalty}\n")
            for cname in ["S1: Room Type Mismatch", "S2: Section Daily Overload",
                          "S3: Instructor Daily Overload", "S4: Schedule Gap", "S5: Late Class"]:
                lf.write(f"  {cname}: {soft_groups_log.get(cname, 0)}\n")

        st.success(f"Log saved to: logs/run_{timestamp}.log")

