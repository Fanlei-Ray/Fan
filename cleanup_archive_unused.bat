@echo off
cd /d E:\FL_Personal\openarm_mujoco-master\openarm_mujoco-master

set ARCHIVE=_archive_unused_after_v18
mkdir %ARCHIVE% 2>nul
mkdir %ARCHIVE%\root_files 2>nul
mkdir %ARCHIVE%\scripts_old 2>nul
mkdir %ARCHIVE%\scripts_old_dirs 2>nul

echo ===== archive root old experiment data =====
for %%F in (
  bc_dataset*.npz
  bc_policy*.pt
  bc_v4*.csv
  model_inspect_output.txt
) do (
  if exist "%%F" move "%%F" "%ARCHIVE%\root_files\"
)

if exist ppo_phase_logs move ppo_phase_logs "%ARCHIVE%\"
if exist ppo_phase_models move ppo_phase_models "%ARCHIVE%\"

echo ===== archive duplicated package folders under scripts =====
if exist scripts\__pycache__ rmdir /s /q scripts\__pycache__
if exist scripts\task_planner0 move scripts\task_planner0 "%ARCHIVE%\scripts_old_dirs\"
if exist scripts\openarm_rl_v15_curriculum_full move scripts\openarm_rl_v15_curriculum_full "%ARCHIVE%\scripts_old_dirs\"
if exist scripts\openarm_rl_v16_reward_fix move scripts\openarm_rl_v16_reward_fix "%ARCHIVE%\scripts_old_dirs\"
if exist scripts\openarm_task_planner_v17_rl_fallback move scripts\openarm_task_planner_v17_rl_fallback "%ARCHIVE%\scripts_old_dirs\"

echo ===== archive old standalone scripts =====
for %%F in (
  scripts\analyze_bc_v4_failures.py
  scripts\bimanual_rule_selector_demo.py
  scripts\build_right_bc_dataset.py
  scripts\build_right_bc_direct_dataset.py
  scripts\cleanup_experiments.py
  scripts\collect_bc_data.py
  scripts\collect_bc_data_v2_bad.py
  scripts\collect_right_bc_dataset.py
  scripts\debug_*.py
  scripts\ik_test.py
  scripts\inspect_model.py
  scripts\left_sequence.py
  scripts\openarm_progress_demo.py
  scripts\random_pick_test.py
  scripts\run_hybrid_bc_*.py
  scripts\run_vision_grasp_demo.py
  scripts\sweep_*.py
  scripts\test_*.py
  scripts\train_bc*.py
  scripts\train_right_bc_direct.py
) do (
  if exist "%%F" move "%%F" "%ARCHIVE%\scripts_old\"
)

echo.
echo ===== cleanup done =====
echo Old files moved to: %ARCHIVE%
echo Core folders kept:
echo   scripts\task_planner
echo   scripts\vision
echo   scripts\rl
echo   v2
echo   src
echo   outputs
pause