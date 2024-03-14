add_executable(model_calibration "")
target_sources(
  model_calibration
  PRIVATE
  model_calibration.cpp
)
target_include_directories(
  model_calibration
  PUBLIC
  ${common_include_directory}
)
target_link_libraries(
  model_calibration
  PRIVATE
  libinroluwb
)

add_executable(parse_colmap_result "")
target_sources(
  parse_colmap_result
  PRIVATE
  parse_colmap_result.cpp
)
target_include_directories(
  parse_colmap_result
  PUBLIC
  ${common_include_directory}
)
target_link_libraries(
  parse_colmap_result
  PRIVATE
  libinroluwb
)

add_executable(match_spline_to_colmap_trajectory "")
target_sources(
  match_spline_to_colmap_trajectory
  PRIVATE
  match_spline_to_colmap_traj.cpp
)
target_include_directories(
  match_spline_to_colmap_trajectory
  PUBLIC
  ${common_include_directory}
)
target_link_libraries(
  match_spline_to_colmap_trajectory
  PRIVATE
  libinroluwb
)

add_executable(scale_matching_trajectory_using_uwb "")
target_sources(
  scale_matching_trajectory_using_uwb
  PRIVATE
  scale_matching_trajectory_using_uwb.cpp
)
target_include_directories(
  scale_matching_trajectory_using_uwb
  PUBLIC
  ${common_include_directory}
)
target_link_libraries(
  scale_matching_trajectory_using_uwb
  PRIVATE
  libinroluwb
)

add_executable(full_batch_optimization "")
target_sources(
  full_batch_optimization
  PRIVATE
  full_batch_optimization.cpp
)
target_include_directories(
  full_batch_optimization
  PUBLIC
  ${common_include_directory}
)
target_link_libraries(
  full_batch_optimization
  PRIVATE
  libinroluwb
)

add_executable(align_estimated_results "")
target_sources(
  align_estimated_results
  PRIVATE
  align_estimated_results.cpp
)
target_include_directories(
  align_estimated_results
  PUBLIC
  ${common_include_directory}
)
target_link_libraries(
  align_estimated_results
  PRIVATE
  libinroluwb
)

add_executable(filtering_state_estimation "")
target_sources(
  filtering_state_estimation
  PRIVATE
  filtering_state_estimation.cpp
)
target_include_directories(
  filtering_state_estimation
  PUBLIC
  ${common_include_directory}
)
target_link_libraries(
  filtering_state_estimation
  PRIVATE
  libinroluwb
)
