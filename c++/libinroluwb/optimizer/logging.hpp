#pragma once

#include <ceres/iteration_callback.h>
#include <ceres/ceres.h>

namespace inrol {
  class LoggingCallback: public ceres::IterationCallback {
  public:
    explicit LoggingCallback(bool log_to_stdout)
        : log_to_stdout_(log_to_stdout) {
    }

    ~LoggingCallback() {
    }

    ceres::CallbackReturnType operator()(
      const ceres::IterationSummary& summary) {
      if (log_to_stdout_) {
        LOG(INFO) << "iter: " << summary.iteration
                  << ", cost: " << summary.cost;
      } else {
        VLOG(1) << "iter: " << summary.iteration << ", cost: " << summary.cost;
      }
      return ceres::SOLVER_CONTINUE;
    }

  private:
    const bool log_to_stdout_;
  };
}  // namespace inrol
