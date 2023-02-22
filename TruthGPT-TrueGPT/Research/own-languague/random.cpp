/// A tensor homothopy in GADT ?

template <typename A, typename B>
struct Homotopy {
  // Definition of constructors and fields goes here
};

template <typename A>
struct Path {
  A start;
  A finish;
};

template <typename A, typename B>
struct Tensor {
  virtual ~Tensor() = default;
};

template <typename A>
struct Scalar : public Tensor<A, A> {
  A value;
};

template <typename A, typename B>
struct Id : public Homotopy<A, A> {};

template <typename A, typename B>
struct Const : public Homotopy<A, B> {
  B value;
};

template <typename A, typename B, typename C>
struct Comp : public Homotopy<A, C> {
  std::unique_ptr<Homotopy<A, B>> f;
  std::unique_ptr<Homotopy<B, C>> g;
};

template <typename A, typename B>
struct Inv : public Homotopy<B, A> {
  std::unique_ptr<Homotopy<A, B>> f;
};

template <typename A, typename B, typename C>
struct Pointwise : public Homotopy<A, C> {
  std::unique_ptr<Homotopy<B, C>> f;
  A x;
};

template <typename A, typename B>
struct Lift : public Homotopy<A, B> {
  std::unique_ptr<Homotopy<A, B>> f;
  std::function<A(const B&)> lift;
};

template <typename A>
struct PathTensor : public Tensor<A, A> {
  std::unique_ptr<Path<A>> path;
};

template <typename A, typename B, typename C>
struct Compose : public Homotopy<A, C> {
  std::function<Homotopy<B, C>(const A&)> f;
  std::function<Homotopy<C, D>(const B&)> g;
  std::function<Homotopy<A, B>(const D&)> h;
};

template <typename A, typename B, typename C, typename D, typename E, typename F>
struct Product : public Homotopy<A, std::tuple<B, D, F>> {
  std::function<Homotopy<B, C>(const A&)> f;
  std::function<Homotopy<D, E>(const A&)> g;
};
