�	�yrM�$6@�yrM�$6@!�yrM�$6@	ej�� ��?ej�� ��?!ej�� ��?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:�yrM�$6@�"���?As�ѓ�5@Y���N�?rEagerKernelExecute 0*	�Q���^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat
i�A'��?!��?��A@)�s�v�4�?1��;��@@:Preprocessing2U
Iterator::Model::ParallelMapV2���;�?!�F��h5@)���;�?1�F��h5@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipn�ݳ��?!5�r�3#Q@)H�`���?1�	V�r�*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate&W��Ma�?!pb�0�0@)���:8؋?1�o�&@:Preprocessing2F
Iterator::ModelW=`2�?!,�5�0s?@)0�1"Qh�?1��=�$@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceS{m��}?![���@)S{m��}?1[���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap_)�Ǻ�?!eF#uË3@)ɑ����j?1��$�-@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor3R臭�d?!_r �� @)3R臭�d?1_r �� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9ej�� ��?I�b%�c�X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�"���?�"���?!�"���?      ��!       "      ��!       *      ��!       2	s�ѓ�5@s�ѓ�5@!s�ѓ�5@:      ��!       B      ��!       J	���N�?���N�?!���N�?R      ��!       Z	���N�?���N�?!���N�?b      ��!       JCPU_ONLYYej�� ��?b q�b%�c�X@Y      Y@qȫvjz�@@"�	
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb�33.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 