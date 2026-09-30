#include <silt/core/operation.hpp>
#include <silt/op/indexed.hpp>

namespace silt {

//
// Gather
//

template<typename T>
tensor_t<T> gather(const tensor_t<T> src, const index_t ind) {
  return op::gather(src, ind);
}

template<typename T>
tensor_t<T> gather(const view_t<T> src, const index_t ind) {
  return op::gather(src, ind);
}

template EXPORT_SHARED silt::tensor_t<int> silt::gather<int>(const silt::tensor_t<int> src, const index_t ind);
template EXPORT_SHARED silt::tensor_t<float> silt::gather<float>(const silt::tensor_t<float> src, const index_t ind);
template EXPORT_SHARED silt::tensor_t<double> silt::gather<double>(const silt::tensor_t<double> src, const index_t ind);

template EXPORT_SHARED silt::tensor_t<int> silt::gather<int>(const silt::view_t<int> src, const index_t ind);
template EXPORT_SHARED silt::tensor_t<float> silt::gather<float>(const silt::view_t<float> src, const index_t ind);
template EXPORT_SHARED silt::tensor_t<double> silt::gather<double>(const silt::view_t<double> src, const index_t ind);

//
// Scatter
//

template<typename T>
void scatter(tensor_t<T> dst, const tensor_t<T> src, const index_t ind) {
  op::scatter(dst, src, ind);
}

template<typename T>
void scatter(view_t<T> dst, const tensor_t<T> src, const index_t ind) {
  op::scatter(dst, src, ind);
}

template EXPORT_SHARED void silt::scatter<int>(silt::tensor_t<int> dst, const silt::tensor_t<int> src, const index_t ind);
template EXPORT_SHARED void silt::scatter<float>(silt::tensor_t<float> dst, const silt::tensor_t<float> src, const index_t ind);
template EXPORT_SHARED void silt::scatter<double>(silt::tensor_t<double> dst, const silt::tensor_t<double> src, const index_t ind);

template EXPORT_SHARED void silt::scatter<int>(silt::view_t<int> dst, const silt::tensor_t<int> src, const index_t ind);
template EXPORT_SHARED void silt::scatter<float>(silt::view_t<float> dst, const silt::tensor_t<float> src, const index_t ind);
template EXPORT_SHARED void silt::scatter<double>(silt::view_t<double> dst, const silt::tensor_t<double> src, const index_t ind);

} // end of namespace silt
