# CITC University Course Timetabling with ETFCSA-TSD

**Thesis Project** — University of Science and Technology of the Philippines (USTP)
**Department:** College of Information Technology and Computing (CITC)
**Algorithm:** (SUBSTRATE DRIFT CLONAL SELECTION ALGORITHM) SDCSA
**Validation Standard:** ITC 2019-inspired constraint framework

---

## Pipeline Overview

```
CITC SCHED (original)                     Structured_Data (teammate's extract)
        |                                           |
        +-------------------------------------------+
                            |
                      preprocess.py
                            |
                     cleaned/ folder
                  (normalized data)
                            |
            cleaned/Algorithm_Input_Cleaned.xlsx
                            |
                   Upload to app.py (Streamlit)
                            |
                    ETFCSA-TSD Optimizer
                            |
                  +--------------------+
                  |                    |
          Optimized_Schedule.xlsx   ITC 2019 XML
                  |
          validate_schedule.py
          (post-optimization check)
                  |
              logs/ folder
          (run_YYYYMMDD_HHMMSS.log)
```

---

## File Structure

| File/Folder | Purpose |
|-------------|---------|
| **`CITC SCHED/`** | Original department schedule (100 section sheets). **Do not modify.** |
| **`Structured_Data_Algorithm_Input_Refined.xlsx`** | Teammate's original input file. **Do not modify.** |
| **`preprocess.py`** | Cleans and normalizes both source files into `cleaned/` |
| **`cleaned/`** | Output of preprocessing — normalized, ready-to-use data |
| `cleaned/CITC_SCHEDULING_CLEANED.xlsx` | Cleaned schedule (100 sheets, all 755 meetings) |
| `cleaned/Algorithm_Input_Cleaned.xlsx` | Optimizer input (Class_Requirements + Room_List + Section_List + Instructors) |
| **`app.py`** | Streamlit web app — runs ETFCSA-TSD optimization with ITC 2019 constraints |
| **`tsd.py`** | ETFCSA-TSD algorithm implementation (generic optimizer, not modified) |
| **`validate_schedule.py`** | ITC 2019-style validator — checks hard/soft constraints independently |
| **`logs/`** | Auto-generated optimization run logs (one per run) |
| **`preprocessing_documentation.md`** | Documents all data quality issues and normalization decisions |
| **`validation_documentation.md`** | Documents ITC 2019 constraint definitions, online class handling, initial results |
| **`optimization_documentation.md`** | Documents encoding, objective function, parameter tuning, logging |

---

## Step-by-Step Usage

### 1. Preprocessing

```bash
python preprocess.py
```

**What it does:**
- Reads the original `CITC SCHED/` Excel (100 section sheets) and `Structured_Data_Algorithm_Input_Refined.xlsx`
- Normalizes instructor names (29 duplicates), subject codes (5), room names (8)
- Detects online classes (Room is NaN = online, 202 of 755 meetings)
- Preserves all 755 multi-day meetings (original input had collapsed them to 488)
- Outputs cleaned files to `cleaned/` folder

**Details:** See [preprocessing_documentation.md](preprocessing_documentation.md)

### 2. Validation of Initial Schedule

```bash
app_visualized_labeled.py
```

**What it does:**
- Validates the original CITC department schedule against ITC 2019-style constraints
- Establishes a baseline: 380 hard violations (223 room + 144 instructor + 13 section), 245 soft violations, INFEASIBLE

**Details:** See [validation_documentation.md](validation_documentation.md)

### 3. Optimization

```bash
streamlit run app.py
```

**Steps in the app:**
1. Upload `cleaned/Algorithm_Input_Cleaned.xlsx`
2. Adjust algorithm parameters in sidebar (population, max evals, clone rate, rho)
3. Adjust soft constraint weights (or set all to 0 for feasibility-first runs)
4. Click **Run Optimization**
5. **Section 3 — Validation:** Review before-vs-after comparison, hard constraint breakdown (H1/H2/H3 in columns), soft constraint breakdown (S1-S5 with DISABLED indicators if weight=0)
6. **Section 4 — Download:** `Optimized_Schedule.xlsx`, ITC 2019 XML, full validation report
7. A run log is auto-saved to `logs/`

**Details:** See [optimization_documentation.md](optimization_documentation.md)

### 4. Review Logs

After each optimization run, a log file is saved to `logs/run_YYYYMMDD_HHMMSS.log` containing:
- Runtime, final fitness
- Algorithm parameters and soft weights used
- Problem size (classes, variables, rooms, timeslots)
- Validation summary (feasibility, hard/soft breakdown by constraint)

Compare logs across runs to identify the best parameter configuration.

---

## Constraint Summary

### Hard Constraints (must be 0 for feasibility)

| ID | Constraint | Applies To |
|----|-----------|-----------|
| H1 | Room Conflict — no two f2f classes in the same room at the same time | Face-to-face only |
| H2 | Instructor Conflict — no instructor teaches two classes at the same time | All classes |
| H3 | Section Conflict — no section attends two classes at the same time | All classes |

### Soft Constraints (minimize for quality)

| ID | Constraint | Default Weight |
|----|-----------|---------------|
| S1 | Room Type Mismatch (lab subject in non-lab room) | 0.5 |
| S2 | Section Daily Overload (> 5 classes/day) | 0.3 |
| S3 | Instructor Daily Overload (> 5 classes/day) | 0.3 |
| S4 | Schedule Gap (> 2h between consecutive classes) | 0.2 |
| S5 | Late Class (starts at or after 6:00 PM) | 0.1 |

---

## Parameter Tuning Guide

| Parameter | Range | Notes |
|-----------|-------|-------|
| Population (N) | 200-300 | Higher = more diversity, slower per generation |
| Max Evaluations | 150,000-500,000 | More evals = better solutions, longer runtime |
| Clones (n_clones) | 10-15 | More clones = more exploitation of good solutions |
| Rho (decay) | 0.95-0.98 | Lower = faster forgetting, helps escape local optima |

**Recommended strategy:**
1. **Phase 1 (Feasibility):** Set all soft weights to 0. Focus on eliminating hard violations.
2. **Phase 2 (Quality):** Once hard violations = 0, re-enable soft weights and run again.

---

## Dependencies

```bash
pip install streamlit pandas numpy openpyxl xlsxwriter requests
```

---

## Data Summary

| Metric | Value |
|--------|-------|
| Total class meetings | 755 |
| Face-to-face meetings | 553 (73%) |
| Online meetings | 202 (27%) |
| Unique sections | 100 |
| Unique instructors | 170 (after normalization) |
| Unique subjects | 82 (after normalization) |
| Unique rooms | 69 (after normalization) |
| Decision variables | 1,308 (553x2 + 202x1) |
| Timeslots | 60 (6 days x 10 periods) |
