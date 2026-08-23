from bevformer.data.nuscenes_categories import (
    CLASS_TO_ID,
    NUSCENES_CLASSES,
    category_to_detection_class,
)


def test_known_categories_map_to_expected_classes():
    assert category_to_detection_class("vehicle.car") == "car"
    assert category_to_detection_class("human.pedestrian.adult") == "pedestrian"
    assert category_to_detection_class("movable_object.trafficcone") == "traffic_cone"


def test_unknown_category_maps_to_none():
    assert category_to_detection_class("static_object.bicycle_rack") is None


def test_class_to_id_covers_all_classes_contiguously():
    assert set(CLASS_TO_ID.keys()) == set(NUSCENES_CLASSES)
    assert sorted(CLASS_TO_ID.values()) == list(range(len(NUSCENES_CLASSES)))
