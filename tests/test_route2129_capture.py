import json
from types import SimpleNamespace as Obj

import torch

from lead.inference.route2129_capture import build_route2129_capture_record


class Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def distance(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class Actor:
    def __init__(self, actor_id, x, type_id="vehicle.car"):
        self.id = actor_id
        self.type_id = type_id
        self.location = Vec(x, 0)
        self.bounding_box = Obj(location=Vec(1, 0), extent=Vec(2, 1, 1), rotation=Obj(yaw=3))

    def get_transform(self):
        location = self.location
        return Obj(
            location=location,
            rotation=Obj(yaw=90),
            transform=lambda offset: Vec(location.x + offset.x, location.y + offset.y),
        )

    def get_velocity(self):
        return Vec(3, 0)

    def get_control(self):
        return Obj(steer=0.1, throttle=0.2, brake=0.0)


def test_route2129_capture_records_models_and_actual_actors_without_mutating():
    ego = Actor(1, 0)
    near = Actor(2, 12, "vehicle.firetruck")
    far = Actor(3, 100)
    world = Obj(get_actors=lambda: [ego, near, far], get_snapshot=lambda: Obj(frame=222))
    prediction = Obj(
        pred_bounding_box_vehicle_system=[Obj(
            x=10, y=1, w=4, h=2, yaw=0.1, velocity=2, brake=0,
            clazz=1, score=0.9,
        )],
        raw_target_speed_scalar=torch.tensor([[5.0]]),
        pred_target_speed_scalar=torch.tensor([[5.0]]),
        pred_route=torch.tensor([[[0.0, 0.0], [5.0, 0.0]]]),
        pred_trajectory=torch.tensor([[[1.0, 0.0]]]),
        route_steer=0.1,
        target_speed_throttle=0.3,
        target_speed_brake=0.0,
    )
    control = Obj(steer=0.1, throttle=0.3, brake=0.0)
    record = build_route2129_capture_record(
        step=212, world=world, ego=ego, prediction=prediction,
        control=control, sensor_speed_mps=4.0,
    )
    assert record["step"] == 212
    assert record["carla_frame"] == 222
    assert record["selected_path_ego_m"] == [[0.0, 0.0], [5.0, 0.0]]
    assert record["predicted_boxes_ego"][0]["score"] == 0.9
    assert [actor["id"] for actor in record["actors_world"]] == [2]
    assert record["actors_world"][0]["box_center_world_m"] == [13.0, 0.0, 0.0]
    assert record["commanded_control"]["throttle"] == 0.3
    assert prediction.pred_target_speed_scalar.item() == 5.0
    json.dumps(record, allow_nan=False)
