python -m grpc_tools.protoc -I protos --python_out=protos --pyi_out=protos --grpc_python_out=protos protos/embedding_service.proto

python -m grpc_tools.protoc -I protos --python_out=protos --pyi_out=protos --grpc_python_out=protos protos/mem.proto