# Paddle预测debug数据生成工具

## 使用说明
`TensorInfo`包含了tensor的描述信息
* name: 名称
* shape: 单个样本的shape
* dtype: tensor的类型，`float32`, `int64`等
* lod\_level: lod\_level


`Tensor`中包含了具体每个样本的数值和lod信息
* data: 样本转为tensor之后的数值
* dtype: 类型，与`TensorInfo`对应
* lod: 相对的lod索引，例如某个样本包含三个句子，每个句子分别包括3, 1, 2个单词，lod为[[3, 1, 2]]


更多使用参考`unittest`

## unittest
运行所有测试
> ```
> python3 -m unittest discover -s tests
> ```

运行单个测试
> ```
> python3 -m unittest tests/test_tensor.py
> ```
