load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "util",
    srcs = [
        "microservices_optimizer.py",
        "complementary_optimizer.py",
        "advanced_complementary_optimizer.py",
        "enhanced_optimizer.py",
    ],
    deps = [
        "//core/common_runtime:common_runtime",
        "//core/framework:framework",
        "//core/platform:platform",
    ],
    visibility = ["//visibility:public"],
)

