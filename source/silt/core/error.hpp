#ifndef SILT_ERROR
#define SILT_ERROR

#include <silt/silt.hpp>
#include <silt/core/types.hpp>
#include <sstream>

// Custom Soillib Exceptions

namespace silt {
namespace error {

#ifdef HAS_CUDA
#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
#endif

template<typename From, typename To>
struct cast_error: std::exception {
  static std::string value() noexcept {
    return std::format("invalid cast from <{}> to <{}>", typedesc<From>::name, typedesc<To>::name);
  }
  const char *what() const noexcept override {
    return value().c_str();
  }
};

// Mismatch Errors

struct mismatch_size: std::exception {
  mismatch_size(size_t want, size_t have) {
    std::stringstream ss;
    ss << "mismatch size. want lhs(";
    ss << want;
    ss << "), have rhs(";
    ss << have;
    ss << ")";
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }

private:
  std::string msg;
};

struct mismatch_type: std::exception {
  mismatch_type(dtype want, dtype have) {
    std::stringstream ss;
    ss << "mismatch type. want (";
    ss << silt::select(want, []<typename S>() { return typedesc<S>::name; });
    ss << "), have (";
    ss << silt::select(have, []<typename S>() { return typedesc<S>::name; });
    ss << ")";
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }

private:
  std::string msg;
};

struct mismatch_host: std::exception {
  mismatch_host(host_t want, host_t have) {
    std::stringstream ss;
    ss << "mismatched host. want <";
    ss << silt::select(want, []<host_t S>() { return hostdesc<S>::name; });
    ss << ">, have <";
    ss << silt::select(have, []<host_t S>() { return hostdesc<S>::name; });
    ss << ">";
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }

private:
  std::string msg;
};

struct unsupported_host: std::exception {
  unsupported_host(host_t want, host_t have) {
    std::stringstream ss;
    ss << "operation not supported for host. want <";
    ss << silt::select(want, []<host_t S>() { return hostdesc<S>::name; });
    ss << ">, have <";
    ss << silt::select(have, []<host_t S>() { return hostdesc<S>::name; });
    ss << ">";
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }

private:
  std::string msg;
};

struct missing_file: std::exception {
  missing_file(std::string file) {
    std::stringstream ss;
    ss << "File does not exist. ";
    ss << file;
    this->msg = ss.str();
  }
  const char *what() const noexcept override {
    return this->msg.c_str();
  }

private:
  std::string msg;
};

} // end of namespace error
} // end of namespace silt

#endif