def test_imports():
    import vision_benchmark_pro
    from vision_benchmark_pro.data import voc_to_yolo
    assert hasattr(voc_to_yolo, 'build_yolo_dataset')
