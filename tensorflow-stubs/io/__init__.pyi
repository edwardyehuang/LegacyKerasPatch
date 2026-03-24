"""Type stubs for tensorflow.io module - provided by LegacyKerasPatch."""

from typing import Any

from . import gfile as gfile

from tensorflow.python.ops.parsing_config import FixedLenFeature as FixedLenFeature
from tensorflow.python.ops.parsing_config import FixedLenSequenceFeature as FixedLenSequenceFeature
from tensorflow.python.ops.parsing_config import VarLenFeature as VarLenFeature
from tensorflow.python.ops.parsing_config import RaggedFeature as RaggedFeature
from tensorflow.python.ops.parsing_config import SparseFeature as SparseFeature

from tensorflow.python.lib.io.tf_record import TFRecordCompressionType as TFRecordCompressionType
from tensorflow.python.lib.io.tf_record import TFRecordOptions as TFRecordOptions
from tensorflow.python.lib.io.tf_record import TFRecordWriter as TFRecordWriter

from tensorflow.python.ops.gen_string_ops import decode_base64 as decode_base64
from tensorflow.python.ops.gen_parsing_ops import decode_compressed as decode_compressed
from tensorflow.python.ops.parsing_ops import decode_csv_v2 as decode_csv
from tensorflow.python.ops.image_ops_impl import decode_image as decode_image
from tensorflow.python.ops.parsing_ops import decode_json_example as decode_json_example
from tensorflow.python.ops.gen_decode_proto_ops import decode_proto_v2 as decode_proto
from tensorflow.python.ops.parsing_ops import decode_raw as decode_raw
from tensorflow.python.ops.sparse_ops import deserialize_many_sparse as deserialize_many_sparse
from tensorflow.python.ops.gen_string_ops import encode_base64 as encode_base64
from tensorflow.python.ops.gen_encode_proto_ops import encode_proto as encode_proto
from tensorflow.python.ops.gen_image_ops import extract_jpeg_shape as extract_jpeg_shape
from tensorflow.python.ops.gen_io_ops import matching_files as matching_files
from tensorflow.python.training.input import match_filenames_once as match_filenames_once
from tensorflow.python.ops.parsing_ops import parse_example_v2 as parse_example
from tensorflow.python.ops.parsing_ops import parse_sequence_example as parse_sequence_example
from tensorflow.python.ops.parsing_ops import parse_single_example_v2 as parse_single_example
from tensorflow.python.ops.parsing_ops import parse_single_sequence_example as parse_single_sequence_example
from tensorflow.python.ops.gen_parsing_ops import parse_tensor as parse_tensor
from tensorflow.python.ops.io_ops import read_file as read_file
from tensorflow.python.ops.sparse_ops import serialize_many_sparse_v2 as serialize_many_sparse
from tensorflow.python.ops.sparse_ops import serialize_sparse_v2 as serialize_sparse
from tensorflow.python.ops.io_ops import serialize_tensor as serialize_tensor
from tensorflow.python.ops.gen_io_ops import write_file as write_file
from tensorflow.python.framework.graph_io import write_graph as write_graph

def __getattr__(name: str) -> Any: ...
