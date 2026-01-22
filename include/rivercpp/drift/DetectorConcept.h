# ifndef DETECTOR_CONCEPT_H
# define DETECTOR_CONCEPT_H

# include <concepts>
# include <utility> // for std::move

namespace rivercpp {
template <typename D>
concept IsDetector = requires(D detector, double value) {
    { detector.update(value) } -> std::same_as<void>;
    { detector.drift_detected } -> std::convertible_to<bool>;
    detector.drift_detected = true;

    requires std::movable<D>;
};

template <IsDetector D, auto ... Args>
requires IsDetector<D> && requires { D(Args...); }
struct DetectorFactory {
    using DetectorType = D;
    static D create() { return D(Args...); }
};

template <typename F>
concept IsDetectorFactory = requires {
    typename F::DetectorType;
    
    { F::create() } -> std::same_as<typename F::DetectorType>;
    
    requires IsDetector<typename F::DetectorType>;
};
}

# endif