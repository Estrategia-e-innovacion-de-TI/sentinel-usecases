def test_imports():
    import sentinel
    import sentinel.detectors
    import sentinel.explorer
    import sentinel.visualization

    assert sentinel.__name__ == "sentinel"
