.PHONY: build
cpp:
	cmake -DCMAKE_BUILD_TYPE=Release -Bbuild cpp/consensus/
	cmake --build build -j$(nproc --all)

.PHONY: demo
demo:
	(cd build && ./tomographic_map_matching_app --data_config ../data/data-demo.json --parameter_config ../data/parameters-consensus.json)
