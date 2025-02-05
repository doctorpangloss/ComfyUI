"""
This should be imported before entrypoints to correctly configure global options prior to importing packages like torch and cv2.

Use this instead of cli_args to import the args:

>>> from comfy.cmd.main_pre import args

It will enable command line argument parsing. If this isn't desired, you must author your own implementation of these fixes.
"""
import ctypes
import importlib.util
import logging
import os
import shutil
import sys
import warnings

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.aio_pika import AioPikaInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.semconv.resource import ResourceAttributes as ResAttrs

from .. import options
from ..app import logger
from ..tracing_compatibility import ProgressSpanSampler
from ..tracing_compatibility import patch_spanbuilder_set_channel
from ..vendor.aiohttp_server_instrumentation import AioHttpServerInstrumentor

this_logger = logging.getLogger(__name__)

options.enable_args_parsing()
if os.name == "nt":
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention.")
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")
warnings.filterwarnings('ignore', category=FutureWarning, message=r'`torch\.cuda\.amp\.custom_fwd.*')

from ..cli_args import args

if args.cuda_device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
    os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
    this_logger.info("Set cuda device to: {}".format(args.cuda_device))

if args.deterministic:
    if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"

if args.oneapi_device_selector is not None:
    os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
    this_logger.info("Set oneapi device selector to: {}".format(args.oneapi_device_selector))


try:
    from . import cuda_malloc
except Exception:
    pass

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
os.environ["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"

def _fix_pytorch_240():
    """Fixes pytorch 2.4.0"""
    torch_spec = importlib.util.find_spec("torch")
    for folder in torch_spec.submodule_search_locations:
        lib_folder = os.path.join(folder, "lib")
        test_file = os.path.join(lib_folder, "fbgemm.dll")
        dest = os.path.join(lib_folder, "libomp140.x86_64.dll")
        if os.path.exists(dest):
            break

        try:
            with open(test_file, 'rb') as f:
                contents = f.read()
                # todo: dubious
                if b"libomp140.x86_64.dll" not in contents:
                    break
            try:
                _ = ctypes.cdll.LoadLibrary(test_file)
            except FileNotFoundError:
                this_logger.warning("Detected pytorch version with libomp issue, trying to patch")
                try:
                    shutil.copyfile(os.path.join(lib_folder, "libiomp5md.dll"), dest)
                except Exception as exc_info:
                    this_logger.error("While trying to patch a fix for torch 2.4.0, an error occurred, which means this is unlikely to work", exc_info=exc_info)
        except:
            pass


def _create_tracer():
    resource = Resource.create({
        ResAttrs.SERVICE_NAME: args.otel_service_name,
        ResAttrs.SERVICE_VERSION: args.otel_service_version,
    })

    # omit progress spans from aio pika
    sampler = ProgressSpanSampler()
    provider = TracerProvider(resource=resource, sampler=sampler)

    is_debugging = hasattr(sys, 'gettrace') and sys.gettrace() is not None
    has_endpoint = args.otel_exporter_otlp_endpoint is not None

    if has_endpoint:
        otlp_exporter = OTLPSpanExporter()
    # elif is_debugging:
    #     otlp_exporter = ConsoleSpanExporter("comfyui")
    else:
        otlp_exporter = SpanExporter()

    processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    # enable instrumentation
    patch_spanbuilder_set_channel()
    AioPikaInstrumentor().instrument()
    AioHttpServerInstrumentor().instrument()
    RequestsInstrumentor().instrument()
    return trace.get_tracer(args.otel_service_name)


def _configure_logging():
    logging_level = args.logging_level
    if args.distributed_queue_worker or args.distributed_queue_frontend or args.distributed_queue_connection_uri is not None:
        logging.basicConfig(level=logging_level)
    else:
        logger.setup_logger(logging_level)


_configure_logging()
_fix_pytorch_240()
tracer = _create_tracer()
__all__ = ["args", "tracer"]
