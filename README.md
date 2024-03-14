# INRoL UWB-based Localization

## Reference 

> T. Kim, B. Yoon, and D. J. Lee, "UWB-Based Localization System Considering Antenna Anisotropy and NLOS/Multipath Conditions," Submitted to IROS 2024

## Environment

To setup the environment, run the following script in the command line:
```
git clone https://github.com/INRoL/inrol_uwb_localization
cd inrol_uwb_localization
pip install -r requirements.txt
```
**Prerequisites**

Install [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), [Ceres Solver](http://ceres-solver.org/installation.html) and [COLMAP](https://colmap.github.io/install.html).

**Build**

```
./script/00_configure_c++_project.sh
./script/00_build_c++_project.sh
```

- - -

# Example

## Dataset

This repository includes the following example datasets in `example/`:

1. `model_calibration/`: Measurement model calibration dataset

2. `model_test/`: Measurement model calibration result test dataset

3. `anchor_self_calibration1/`: Anchor self-calibration dataset (env#1)

4. `anchor_self_calibration2/`: Anchor self-calibration dataset (env#2)

5. `filtering_state_estimation1/`: Filtering-based state estimation dataset (env#1)

6. `filtering_state_estimation2/`: Filtering-based state estimation dataset (env#2)

## 1. Measurement Model Calibration

1. Run calibration

* Model A (not using directional bias, Gaussian noise model)

```
./script/01-1_model_calibration_A.sh
```

* Model B (using directional bias, Gaussian noise model)

```
./script/01-2_model_calibration_B.sh
```

* Model C (using directional bias, asymmetric heavy-tailed noise model)

```
./script/01-3_model_calibration_C.sh
```

2. Evaluate the calibration results

```
./script/01-4_plot_uwb_error_histogram.sh
```

## 2. Anchor Self-Calibration 

1. Create colmap project

```
./script/02-1_create_colmap_prj.sh
```

2. Run colmap

```
colmap database_creator --database_path example/anchor_self_calibration1/02-1_create_colmap_prj/database.db
colmap feature_extractor --project_path example/anchor_self_calibration1/02-1_create_colmap_prj/feature_extractor_config.ini
colmap sequential_matcher --project_path example/anchor_self_calibration1/02-1_create_colmap_prj/sequential_matcher_config.ini
colmap mapper --project_path example/anchor_self_calibration1/02-1_create_colmap_prj/mapper_config.ini
```

3. Parse colmap result

```
./script/02-2_parse_colmap_result.sh
```

4. Match B-spline to colmap result

```
./script/02-3_match_bspline_to_colmap_traj.sh
```

5. Find scale of B-spline

```
./script/02-4_scale_matching_traj.sh
```

6. Full batch optimization

* Model A (not using directional bias, Gaussian noise model)

```
./script/02-5_full_batch_optimization_A.sh
```

* Model B (using directional bias, Gaussian noise model)

```
./script/02-6_full_batch_optimization_B.sh
```

* Model C (using directional bias, asymmetric heavy-tailed noise model)

```
./script/02-7_full_batch_optimization_C.sh
```

## 3. Filtering-based state estimation

* Model A, Standard EKF

```
./script/03-1_kalman_filter_A.sh
```

* Model A, Huber-norm based update

```
./script/03-2_kalman_filter_AH.sh
```

* Model B, Standard EKF

```
./script/03-3_kalman_filter_B.sh
```

* Model B, Huber-norm based update

```
./script/03-4_kalman_filter_BH.sh
```

* Model C, Proposed method

```
./script/03-5_kalman_filter_C.sh
```


## Credits

This repository uses some codes of [RPG vision-based SLAM](https://github.com/uzh-rpg/rpg_vision-based_slam) to utilize COLMAP
