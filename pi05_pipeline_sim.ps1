<#
.SYNOPSIS
  pi05 Fine-tuning Pipeline Simulation (PowerShell, no dependencies)
.DESCRIPTION
  Simulates 200-400 demo episodes through 5 stages:
    1. Data Preprocessing
    2. Dataset Merging
    3. Training Config
    4. Norm Statistics
    5. Training Launch
#>

$ErrorActionPreference = "Stop"
$BASE_DIR = $PSScriptRoot
if (-not $BASE_DIR) { $BASE_DIR = Get-Location }

$script:NUM_EPISODES = Get-Random -Minimum 200 -Maximum 401
$FRAMES_PER_EPISODE = 100
$JOINT_DIM = 14
$GRIPPER_DIM = 2
$TORSO_DIM = 6
$STATE_DIM = $TORSO_DIM + $JOINT_DIM + $GRIPPER_DIM

$script:LOG = New-Object System.Collections.ArrayList
function Log([string]$msg = "") {
    $script:LOG.Add($msg) | Out-Null
    Write-Host $msg
}

function Get-Gaussian([double]$mean = 0, [double]$stddev = 1) {
    $u1 = (Get-Random -Minimum 1 -Maximum 10000) / 10000.0
    $u2 = (Get-Random -Minimum 1 -Maximum 10000) / 10000.0
    $z = [math]::Sqrt(-2 * [math]::Log($u1)) * [math]::Cos(2 * [math]::PI * $u2)
    return $mean + $stddev * $z
}

function Clamp([double]$v, [double]$lo, [double]$hi) {
    return [math]::Max($lo, [math]::Min($hi, $v))
}

# ====================================================================
#  Stage 1: Data Preprocessing
# ====================================================================
function Stage1-Preprocess {
    Log ""
    Log ("=" * 60)
    Log "[Stage 1 - Data Preprocessing]"
    Log "  Script: scripts/preprocess_lerobot_data_fast.py"
    Log ("=" * 60)
    Log "  Demo episodes: $script:NUM_EPISODES"
    Log "  Frames per episode: $FRAMES_PER_EPISODE"
    Log "  State dim: $STATE_DIM (torso $TORSO_DIM + joints $JOINT_DIM + grippers $GRIPPER_DIM)"

    $results = @{}

    $tasks = @(
        @{ name = "water_hose_insertion"; ee = $false },
        @{ name = "cable_connect_mainboard"; ee = $true }
    )

    foreach ($task in $tasks) {
        $taskName = $task.name
        $eeMode = $task.ee
        $prefix = if ($eeMode) { "ee_filted" } else { "filted" }
        $outName = "lerobot_260710_${taskName}_v5_${prefix}_rby1_$($script:NUM_EPISODES)s"
        $outPath = Join-Path $BASE_DIR "mock_processed\$outName"

        $modeStr = if ($eeMode) { "EE" } else { "Joint" }
        Log ""
        Log "  --- Task: $taskName | Mode: $modeStr ---"
        Log "  Output: $outPath"

        $totalFrames = $script:NUM_EPISODES * $FRAMES_PER_EPISODE
        $effDim = if ($eeMode) { $STATE_DIM - $TORSO_DIM } else { $STATE_DIM }
        # Note: In real script, each frame's state/action is processed.
        # Here we simulate the statistics without per-frame loops for speed.
        Log "      Processing $totalFrames frames (simulated)..."


        foreach ($sub in @("data", "videos", "meta")) {
            $dir = Join-Path $outPath $sub
            if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        }

        $meta = @{
            total_episodes = $script:NUM_EPISODES
            total_frames = $totalFrames
            frames_per_episode = $FRAMES_PER_EPISODE
            state_dim = $effDim
            action_dim = $effDim
            ee_mode = $eeMode
            task = $taskName
        } | ConvertTo-Json -Depth 3
        Set-Content -Path (Join-Path $outPath "meta\info.json") -Value $meta -Encoding UTF8

        # Write sample episode
        $sample = @()
        for ($t = 0; $t -lt $FRAMES_PER_EPISODE; $t++) {
            $s = @()
            for ($i = 0; $i -lt $effDim; $i++) { $s += [math]::Round((Get-Gaussian 0 0.3), 4) }
            $sample += ,@{ state = $s; action = $s }
        }
        $sampleJson = $sample | ConvertTo-Json -Depth 3
        Set-Content -Path (Join-Path $outPath "data\episode_000000.json") -Value $sampleJson -Encoding UTF8

        Log "  [OK] Done! total_frames=$totalFrames, state_dim=$effDim"
        Log "      data/ videos/ meta/ subdirectories created"

        $results[$taskName] = $outPath
    }

    Log ""
    Log "  [Key Points - Stage 1]"
    Log "    1. fix_outliers        -> clip outliers to [-3, 3]"
    Log "    2. action_to_next_state-> action=next frame state, smooth trajectory"
    Log "    3. ee_mode             -> EE space conversion (mainboard), else joint space"
    Log "    4. --output_dir required, else modifies raw data irreversibly"
    Log "    5. Naming: insert 'filted' or 'ee_filted' before 'rby1'"

    return $results
}

# ====================================================================
#  Stage 2: Merge Datasets
# ====================================================================
function Stage2-Merge($processedPaths) {
    Log ""
    Log ("=" * 60)
    Log "[Stage 2 - Merge Datasets]"
    Log "  Script: scripts/merge_lerobot_fast.py"
    Log ("=" * 60)

    $waterPath = $processedPaths["water_hose_insertion"]
    $mergedEp = $script:NUM_EPISODES * 2
    $tgtName = "lerobot_260710_water_hose_insertion_v5_merged_rby1_${mergedEp}s"
    $tgtPath = Join-Path $BASE_DIR "mock_merged\$tgtName"

    Log "  Source 1: $(Split-Path $waterPath -Leaf)"
    Log "  Source 2: (simulated 2nd batch)"
    Log "  Target: $tgtPath"

    $meta1 = Get-Content (Join-Path $waterPath "meta\info.json") -Raw | ConvertFrom-Json
    $totalEp = $meta1.total_episodes * 2
    $totalFr = $meta1.total_frames * 2

    foreach ($sub in @("data", "videos", "meta")) {
        $dir = Join-Path $tgtPath $sub
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    }

    $mergedMeta = @{
        total_episodes = $totalEp
        total_frames = $totalFr
        state_dim = $meta1.state_dim
        action_dim = $meta1.action_dim
        sub_datasets = @("v4_filted", "v5_filted")
        repo_id = "merged_water_hose"
    } | ConvertTo-Json -Depth 3
    Set-Content -Path (Join-Path $tgtPath "meta\info.json") -Value $mergedMeta -Encoding UTF8

    Log ""
    Log "  Merged episodes: $totalEp"
    Log "  Merged frames: $totalFr"
    Log "  [OK] Verified: $totalEp == $($meta1.total_episodes) + $($meta1.total_episodes)"

    Log ""
    Log "  [Key Points - Stage 2]"
    Log "    1. --tgt_path must not exist beforehand, script creates it"
    Log "    2. Verify merged count == sum of subsets"
    Log "    3. Can skip if only one batch of data"

    return @{
        water_hose_merged = $tgtPath
        mainboard_merged = $processedPaths["cable_connect_mainboard"]
    }
}

# ====================================================================
#  Stage 3: Write Training Config
# ====================================================================
function Stage3-Config($mergedPaths) {
    Log ""
    Log ("=" * 60)
    Log "[Stage 3 - Write Training Config]"
    Log "  File: src/openpi/training/config.py -> TrainConfig"
    Log ("=" * 60)

    $mergedEp = $script:NUM_EPISODES * 2

    # Water hose (joint + RTC)
    $waterName = "pi05_rby1_water_hole_insertion_v5_mengfan_${mergedEp}s_from_base_bimodal_rtc_delta_absgripper_260710"
    $waterCfg = [ordered]@{
        name = $waterName
        exp_name = "finetune"
        task = "water_hose_insertion"
        control_mode = "joint"
        rtc_enabled = $true
        model = [ordered]@{
            action_horizon = 30
            pi05 = $true
            max_delay = 8
            delay_sampling = "bimodal"
            delay_sampling_temperature = 1.0
            delay_sampling_second_peak = 6
            delay_sampling_second_peak_width = 1.5
            delay_sampling_second_peak_weight = 0.5
            rtc_loss_scale_mode = "batch"
            rtc_loss_scale_cap = $null
        }
        data = [ordered]@{
            repo_id = $mergedPaths["water_hose_merged"]
            arm_joint_mask = "make_bool_mask(7, -7, 1, -1)"
            default_prompt = "insert the right water hose into the hole"
            exclude_torso = $true
            cameras = @("cam_high_left", "cam_high_right", "cam_left_wrist", "cam_right_wrist")
        }
        assets = [ordered]@{
            assets_dir = (Split-Path $mergedPaths["water_hose_merged"] -Parent)
            asset_id = (Split-Path $mergedPaths["water_hose_merged"] -Leaf)
        }
        weight_loader = @{ path = "/tmp/pi05_base/params/" }
        num_train_steps = 100000
        keep_period = 10000
        lr_schedule = [ordered]@{
            warmup_steps = 1000
            peak_lr = 2e-5
            decay_steps = 50000
            decay_lr = 2e-6
        }
        batch_size = 32
        log_interval = 50
        num_workers = 16
        fsdp_devices = 8
    }

    # Mainboard (EE, no RTC)
    $mbName = "pi05_rby1_cable_connect_mainboard_v4_$($script:NUM_EPISODES)s_from_base_ee_delta_euler_260710"
    $mbCfg = [ordered]@{
        name = $mbName
        exp_name = "finetune"
        task = "cable_connect_mainboard"
        control_mode = "ee"
        rtc_enabled = $false
        model = [ordered]@{
            action_dim = 32
            action_horizon = 30
            pi05 = $true
        }
        data = [ordered]@{
            repo_id = $mergedPaths["mainboard_merged"]
            default_prompt = "cable mainboard insertion"
            action_space = "ee"
            ee_pose_repr = "euler"
            arm_mode = "left"
            use_delta_actions = $true
            use_cam_high_right = $false
        }
        assets = [ordered]@{
            assets_dir = (Split-Path $mergedPaths["mainboard_merged"] -Parent)
            asset_id = (Split-Path $mergedPaths["mainboard_merged"] -Leaf)
        }
        weight_loader = @{ path = "/tmp/pi05_base/params/" }
        num_train_steps = 100000
        keep_period = 10000
        lr_schedule = [ordered]@{
            warmup_steps = 1000
            peak_lr = 2e-5
            decay_steps = 50000
            decay_lr = 2e-6
        }
        batch_size = 32
        log_interval = 50
        num_workers = 16
        fsdp_devices = 8
    }

    $configs = @{ water_hose = $waterCfg; mainboard = $mbCfg }
    $cfgPath = Join-Path $BASE_DIR "training_configs.json"
    $configs | ConvertTo-Json -Depth 6 | Set-Content -Path $cfgPath -Encoding UTF8

    Log ""
    Log "  [OK] Water hose config:"
    Log "      name: $waterName"
    Log "      joint+RTC | right arm | action_horizon=30 | steps=100000"
    Log "      arm_joint_mask: make_bool_mask(7,-7,1,-1) -> left arm/gripper masked"
    Log "      RTC: bimodal, max_delay=8"

    Log ""
    Log "  [OK] Mainboard config:"
    Log "      name: $mbName"
    Log "      EE(no RTC) | left arm | action_dim=32 | action_space=ee"
    Log "      ee_pose_repr=euler | use_delta_actions=True"
    Log "      arm_mode=left -> right arm masked"

    Log ""
    Log "  Configs saved: $cfgPath"

    Log ""
    Log "  [Key Points - Stage 3]"
    Log "    1. Config name must match --config-name exactly"
    Log "    2. Water hose = joint+RTC (right arm); Mainboard = EE (left arm, no RTC)"
    Log "    3. assets points to norm: assets_dir/asset_id/norm_stats.json"
    Log "    4. from base: 100k steps/save every 10k; from ckpt: 20k steps/save every 1k"
    Log "    5. decay_steps = num_train_steps / 2"
    Log "    6. Naming: pi05_robot_task_version_Ns_from_source_features_date"

    return $configs
}

# ====================================================================
#  Stage 4: Compute Norm Statistics
# ====================================================================
function Stage4-Norm($configs) {
    Log ""
    Log ("=" * 60)
    Log "[Stage 4 - Compute Norm Statistics]"
    Log "  Script: scripts/compute_norm_states_fast.py"
    Log ("=" * 60)

    $normPaths = @{}
    $taskConfigs = @(
        @{ key = "water_hose"; torso = 6 },
        @{ key = "mainboard"; torso = 0 }
    )

    foreach ($tc in $taskConfigs) {
        $taskKey = $tc.key
        $torsoSkip = $tc.torso
        $cfg = $configs[$taskKey]
        $datasetPath = $cfg.data.repo_id
        $metaFile = Join-Path $datasetPath "meta\info.json"
        $meta = Get-Content $metaFile -Raw | ConvertFrom-Json

        $rawDim = $meta.state_dim
        $effDim = $rawDim - $torsoSkip
        $totalFrames = $meta.total_frames

        $torsoStr = if ($torsoSkip -gt 0) { "6 (required for joint mode)" } else { "none (forbidden for EE mode)" }

        Log ""
        Log "  --- $taskKey ---"
        Log "  config: $($cfg.name)"
        Log "  dataset: $(Split-Path $datasetPath -Leaf)"
        Log "  --torso: $torsoStr"
        Log "  raw state_dim: $rawDim, effective: $effDim"
        Log "  total frames: $totalFrames"

        $stateMean = @()
        $stateStd = @()
        for ($i = 0; $i -lt $effDim; $i++) {
            $stateMean += [math]::Round((Get-Random -Minimum -500 -Maximum 500) / 1000.0, 4)
            $stateStd += [math]::Round((Get-Random -Minimum 50 -Maximum 300) / 1000.0, 4)
        }

        $normStats = [ordered]@{
            state = [ordered]@{ mean = $stateMean; std = $stateStd }
            action = [ordered]@{ mean = $stateMean; std = $stateStd }
            num_frames = $totalFrames
            torso_skipped = $torsoSkip
            config_name = $cfg.name
        }

        $normDir = Join-Path $cfg.assets.assets_dir $cfg.assets.asset_id
        if (-not (Test-Path $normDir)) { New-Item -ItemType Directory -Path $normDir -Force | Out-Null }
        $normPath = Join-Path $normDir "norm_stats.json"
        $normStats | ConvertTo-Json -Depth 5 | Set-Content -Path $normPath -Encoding UTF8

        $meanStr = ($stateMean[0..2] | ForEach-Object { $_.ToString("F4") }) -join ", "
        $stdStr = ($stateStd[0..2] | ForEach-Object { $_.ToString("F4") }) -join ", "
        Log "  [OK] norm_stats.json generated: $normPath"
        Log "      state mean[0:3]: $meanStr"
        Log "      state std[0:3]:  $stdStr"

        $normPaths[$taskKey] = $normPath
    }

    Log ""
    Log "  [Key Points - Stage 4]"
    Log "    1. [WARNING] joint mode (water hose) -> MUST add --torso 6, skip 6 torso dims"
    Log "    2. [WARNING] EE mode (mainboard) -> MUST NOT add --torso 6"
    Log "    3. Gripper out of range (0~0.1) -> add --correct (modifies data files)"
    Log "    4. norm_stats.json stored at assets_dir/asset_id/"
    Log "    5. Training and inference MUST use the same norm"
    Log "    6. Can skip if reusing old norm, just point assets correctly"
    Log "    7. [WARNING] Most error-prone step! Wrong --torso corrupts the model!"

    return $normPaths
}

# ====================================================================
#  Stage 5: Launch Training
# ====================================================================
function Get-CosineLR($step, $warmup, $peakLr, $decaySteps, $decayLr) {
    if ($step -lt $warmup) {
        return $peakLr * ($step + 1) / $warmup
    }
    $progress = [math]::Min(($step - $warmup) / [math]::Max($decaySteps - $warmup, 1), 1.0)
    return $decayLr + 0.5 * ($peakLr - $decayLr) * (1 + [math]::Cos([math]::PI * $progress))
}

function Stage5-Train($configs) {
    Log ""
    Log ("=" * 60)
    Log "[Stage 5 - Launch Training]"
    Log "  Script: scripts/train.py"
    Log ("=" * 60)

    $ckptDirs = @{}

    foreach ($taskKey in @("water_hose", "mainboard")) {
        $cfg = $configs[$taskKey]
        $label = if ($taskKey -eq "water_hose") { "Water Hose Insertion (joint+RTC)" } else { "Mainboard Cable (EE, no RTC)" }
        $numSteps = $cfg.num_train_steps
        $keepPeriod = $cfg.keep_period
        $warmup = $cfg.lr_schedule.warmup_steps
        $peakLr = $cfg.lr_schedule.peak_lr
        $decaySteps = $cfg.lr_schedule.decay_steps
        $decayLr = $cfg.lr_schedule.decay_lr

        $ckptBase = Join-Path $BASE_DIR "mock_checkpoints\$($cfg.name)\finetune\0"
        if (-not (Test-Path $ckptBase)) { New-Item -ItemType Directory -Path $ckptBase -Force | Out-Null }

        $dataDesc = if ($taskKey -eq "water_hose") { "$($script:NUM_EPISODES) eps (merged)" } else { "$($script:NUM_EPISODES) eps (single)" }

        Log ""
        Log "  --- $label ---"
        Log "  config:       $($cfg.name)"
        Log "  total steps:  $($numSteps.ToString('N0'))"
        Log "  save period:  every $($keepPeriod.ToString('N0')) steps"
        Log "  batch_size:   $($cfg.batch_size)"
        Log "  init weights: $($cfg.weight_loader.path)"
        Log "  control mode: $($cfg.control_mode)"
        Log "  RTC:          $($cfg.rtc_enabled)"
        Log "  demo data:    $dataDesc"
        Log "  checkpoint:   $ckptBase"

        # Sample key steps
        $demoSteps = [System.Collections.Generic.List[int]]::new()
        for ($s = 0; $s -lt 500; $s += 50) { $demoSteps.Add($s) }
        for ($s = 0; $s -le $numSteps; $s += $keepPeriod) { $demoSteps.Add($s) }
        $demoSteps.Add($numSteps)
        $demoSteps = $demoSteps | Sort-Object -Unique

        # Simulate loss curve
        $lossLog = [System.Collections.ArrayList]::new()
        $initialLoss = 2.0
        $finalLoss = 0.15

        foreach ($step in $demoSteps) {
            $lr = Get-CosineLR $step $warmup $peakLr $decaySteps $decayLr
            if ($step -lt $warmup) {
                $progress = $step / [math]::Max($warmup, 1)
                $baseLoss = $initialLoss - ($initialLoss - 1.0) * $progress * 0.5
            } else {
                $progress = [math]::Min(($step - $warmup) / [math]::Max($numSteps - $warmup, 1), 1.0)
                $baseLoss = 1.0 + ($finalLoss - 1.0) * (0.5 * (1 - [math]::Cos([math]::PI * $progress)))
            }
            $noise = (Get-Gaussian 0 0.01)
            $loss = [math]::Max(0.01, $baseLoss + $noise)
            $lossLog.Add(@($step, [math]::Round($loss, 6), $lr)) | Out-Null
        }

        Log ""
        Log "    step       loss            lr"
        Log "    ----       ----           ----"
        $count = 0
        foreach ($entry in $lossLog) {
            if ($count -lt 12) {
                $s = "{0,6}" -f $entry[0]
                $l = "{0,10:F6}" -f $entry[1]
                $r = "{0,12:E2}" -f $entry[2]
                Log "    $s  $l  $r"
            }
            $count++
        }
        if ($lossLog.Count -gt 17) {
            Log "    ..."
            foreach ($entry in $lossLog[-5..-1]) {
                $s = "{0,6}" -f $entry[0]
                $l = "{0,10:F6}" -f $entry[1]
                $r = "{0,12:E2}" -f $entry[2]
                Log "    $s  $l  $r"
            }
        }

        # Save checkpoints
        $ckptSteps = @()
        for ($cs = $keepPeriod; $cs -le $numSteps; $cs += $keepPeriod) { $ckptSteps += $cs }

        Log ""
        Log "  Saving $($ckptSteps.Count) checkpoints:"
        foreach ($cs in $ckptSteps) {
            $ckptDir = Join-Path $ckptBase $cs.ToString()
            if (-not (Test-Path $ckptDir)) { New-Item -ItemType Directory -Path $ckptDir -Force | Out-Null }

            $ckptLoss = 0.15
            for ($i = $lossLog.Count - 1; $i -ge 0; $i--) {
                if ($lossLog[$i][0] -le $cs) { $ckptLoss = $lossLog[$i][1]; break }
            }
            $ckptMeta = @{ step = $cs; config = $cfg.name; loss = $ckptLoss } | ConvertTo-Json
            Set-Content -Path (Join-Path $ckptDir "checkpoint_meta.json") -Value $ckptMeta -Encoding UTF8

            if ($cs -le ($keepPeriod * 3) -or $cs -ge $numSteps) {
                Log ("     step {0} -> {1}" -f $cs, $ckptDir)
            }
        }
        if ($ckptSteps.Count -gt 6) {
            Log "     ... ($($ckptSteps.Count - 6) intermediate checkpoints)"
        }

        # Write loss.log
        $logPath = Join-Path $ckptBase "loss.log"
        $logLines = @()
        $logLines += "# Training log for $($cfg.name)"
        $logLines += "# Total steps: $numSteps, Batch size: $($cfg.batch_size)"
        $logLines += "    step       loss            lr"
        foreach ($entry in $lossLog) {
            $s = "{0,6}" -f $entry[0]
            $l = "{0,10:F6}" -f $entry[1]
            $r = "{0,12:E2}" -f $entry[2]
            $logLines += "    $s  $l  $r"
        }
        Set-Content -Path $logPath -Value $logLines -Encoding UTF8

        $finalLoss = $lossLog[-1][1]
        $finalStr = "{0:F6}" -f $finalLoss
        Log ""
        Log "  [OK] Training complete!"
        Log "      loss.log: $logPath"
        Log "      final loss: $finalStr"
        Log "      checkpoints: $($ckptSteps.Count)"

        $ckptDirs[$taskKey] = $ckptBase
    }

    Log ""
    Log "  [Key Points - Stage 5]"
    Log "    1. from base: 100k steps/save 10k; from ckpt: 20k steps/save 1k"
    Log "    2. --overwrite clears dir; --resume continues from checkpoint"
    Log "    3. XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 limits GPU memory"
    Log "    4. loss printed every 50 steps, should decrease, no spikes"
    Log "    5. checkpoints at checkpoint_base_dir/config_name/exp_name/step/"
    Log "    6. fsdp_devices=8 -> 8-GPU FSDP parallel training"
    Log "    7. Full command:"
    Log "       XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py <config> --exp-name=0 --overwrite"

    return $ckptDirs
}

# ====================================================================
#  Main
# ====================================================================
Log ("=" * 60)
Log "  pi05 Fine-tuning Pipeline Simulation (PowerShell)"
Log "  Simulating 200-400 demo episodes through full training"
Log "  Tasks: water hose (joint+RTC) + mainboard (EE)"
Log ("=" * 60)
Log ""
Log "Demo episodes this run: $script:NUM_EPISODES"
Log "State dim: $STATE_DIM (torso $TORSO_DIM + joints $JOINT_DIM + grippers $GRIPPER_DIM)"

$processed = Stage1-Preprocess
$merged = Stage2-Merge $processed
$configs = Stage3-Config $merged
$normPaths = Stage4-Norm $configs
$ckptDirs = Stage5-Train $configs

# Save pipeline state
$state = [ordered]@{
    num_episodes = $script:NUM_EPISODES
    water_hose_processed = $processed["water_hose_insertion"]
    mainboard_processed = $processed["cable_connect_mainboard"]
    water_hose_merged = $merged["water_hose_merged"]
    mainboard_merged = $merged["mainboard_merged"]
    water_config_name = $configs["water_hose"].name
    mainboard_config_name = $configs["mainboard"].name
    water_norm_path = $normPaths["water_hose"]
    mainboard_norm_path = $normPaths["mainboard"]
    water_ckpt_dir = $ckptDirs["water_hose"]
    mainboard_ckpt_dir = $ckptDirs["mainboard"]
}
$statePath = Join-Path $BASE_DIR "pipeline_state.json"
$state | ConvertTo-Json -Depth 3 | Set-Content -Path $statePath -Encoding UTF8

Log ""
Log ("=" * 60)
Log "  All 5 stages complete!"
Log ("=" * 60)
Log ""
Log "  Generated artifacts:"
Log "    mock_processed/         Preprocessed datasets"
Log "    mock_merged/            Merged datasets"
Log "    training_configs.json   Training configs for both tasks"
Log "    norm_stats.json         Norm statistics (in dataset dirs)"
Log "    mock_checkpoints/       Training checkpoints + loss.log"
Log "    pipeline_state.json     Pipeline state"

$logPath = Join-Path $BASE_DIR "pipeline_output.log"
$script:LOG -join "`r`n" | Set-Content -Path $logPath -Encoding UTF8
Write-Host ""
Write-Host "Full log saved: $logPath"
