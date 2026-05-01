load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "core",
    srcs = [
        "__init__.py",
    ],
    deps = [
        "//core/common_runtime:common_runtime",
        "//core/data:data",
        "//core/distributed_runtime:distributed_runtime",
        "//core/framework:framework",
        "//core/kernels:kernels",
        "//core/lib:lib",
        "//core/ops:ops",
        "//core/platform:platform",
        "//core/runtime_fallback:runtime_fallback",
        "//core/util:util",
    ],
    visibility = ["//visibility:public"],
)

