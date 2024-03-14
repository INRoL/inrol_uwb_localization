add_library(libinroluwb "")

target_sources(
  libinroluwb
  PRIVATE
  filter/kalman_filter.cpp
  filter/filter_math_utility.cpp
  filter/jacobian.cpp
  geometry/feature.cpp
  geometry/pose.cpp
  optimizer/align.cpp
  optimizer/batch.cpp
  optimizer/calibration.cpp
  optimizer/match.cpp
  optimizer/scale.cpp
  optimizer/math/harmonics.cpp
  utils/binary.cpp
  utils/directory.cpp
  utils/param.cpp
  utils/text.cpp
)

target_include_directories(
  libinroluwb
  PUBLIC
  ${common_include_directory}
)

target_link_libraries(
  libinroluwb
  PUBLIC
  ${CERES_LIBRARIES}
  ${glog_LIBRARIES}
  yaml-cpp
)
