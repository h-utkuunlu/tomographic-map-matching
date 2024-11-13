.PHONY: cpp
cpp:
	cmake -DCMAKE_BUILD_TYPE=Release -Bbuild cpp/tomographic_map_matching/
	cmake --build build -j$(nproc --all)

.PHONY: demo
demo:
	(cd build && ./tomographic_map_matching_app --data_config ../data/data-demo.json --parameter_config ../data/parameters-consensus.json)

# Experimental deployment
IMAGE=cair_unitree_tomographic

.PHONY: run
run:
	@docker pull localhost:5000/$(IMAGE)
	@docker tag localhost:5000/$(IMAGE) $(IMAGE)
	@docker compose run --rm $(IMAGE)

.PHONY: build
build:
	@docker compose build
	@docker tag $(IMAGE) b1-154:5000/$(IMAGE)
	@docker tag $(IMAGE) b1-284:5000/$(IMAGE)

robot_ids = 154 284

$(robot_ids): build
	@docker push b1-$@:5000/$(IMAGE)
