"""Unit tests for src/visualization/plotting.py."""

import base64
import io

import numpy as np
import plotly.graph_objects as go
import pytest

from src.visualization.plotting import (
    DEFECT_CLASS_PALETTE,
    DEFECT_CLASSES,
    apply_default_layout,
    create_hover_template,
    create_scatter_with_class_colors,
    create_wafer_colormap,
    get_class_color,
    wafer_map_to_base64_png,
    _resize_nearest,
)


class TestDefectClassPalette:
    """Tests for the colorblind-friendly palette."""

    def test_palette_has_9_classes(self):
        assert len(DEFECT_CLASS_PALETTE) == 9

    def test_all_expected_classes_present(self):
        expected = {
            "Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Near-full", "Random", "Scratch", "none",
        }
        assert set(DEFECT_CLASS_PALETTE.keys()) == expected

    def test_all_colors_are_hex_strings(self):
        for cls_name, color in DEFECT_CLASS_PALETTE.items():
            assert color.startswith("#"), f"{cls_name} color doesn't start with #"
            assert len(color) == 7, f"{cls_name} color is not 7 chars: {color}"
            # Validate hex characters
            int(color[1:], 16)

    def test_all_colors_are_distinct(self):
        colors = list(DEFECT_CLASS_PALETTE.values())
        assert len(set(colors)) == 9, "All 9 colors must be distinct"

    def test_defect_classes_list_matches_palette_keys(self):
        assert DEFECT_CLASSES == list(DEFECT_CLASS_PALETTE.keys())


class TestGetClassColor:
    """Tests for get_class_color function."""

    def test_known_class_returns_palette_color(self):
        assert get_class_color("Center") == "#4477AA"
        assert get_class_color("Donut") == "#EE6677"

    def test_unknown_class_returns_grey_fallback(self):
        assert get_class_color("UnknownClass") == "#999999"
        assert get_class_color("") == "#999999"


class TestApplyDefaultLayout:
    """Tests for apply_default_layout function."""

    def test_returns_same_figure(self):
        fig = go.Figure()
        result = apply_default_layout(fig, title="Test")
        assert result is fig

    def test_sets_title(self):
        fig = go.Figure()
        apply_default_layout(fig, title="My Title")
        assert fig.layout.title.text == "My Title"

    def test_no_title_when_empty_string(self):
        fig = go.Figure()
        apply_default_layout(fig, title="")
        # Should not set a title when empty string
        assert fig.layout.title.text is None or fig.layout.title.text == ""

    def test_sets_dimensions(self):
        fig = go.Figure()
        apply_default_layout(fig, width=800, height=600)
        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_dark_mode_background(self):
        fig = go.Figure()
        apply_default_layout(fig, dark_mode=True)
        assert fig.layout.paper_bgcolor == "#1e1e1e"
        assert fig.layout.plot_bgcolor == "#1e1e1e"

    def test_light_mode_background(self):
        fig = go.Figure()
        apply_default_layout(fig, dark_mode=False)
        assert fig.layout.paper_bgcolor == "#ffffff"
        assert fig.layout.plot_bgcolor == "#ffffff"

    def test_legend_visibility(self):
        fig = go.Figure()
        apply_default_layout(fig, show_legend=False)
        assert fig.layout.showlegend is False


class TestCreateWaferColormap:
    """Tests for create_wafer_colormap function."""

    def test_shape(self):
        cmap = create_wafer_colormap()
        assert cmap.shape == (3, 3)

    def test_dtype(self):
        cmap = create_wafer_colormap()
        assert cmap.dtype == np.uint8

    def test_black_for_background(self):
        cmap = create_wafer_colormap()
        np.testing.assert_array_equal(cmap[0], [0, 0, 0])

    def test_green_for_normal(self):
        cmap = create_wafer_colormap()
        # Green channel should be highest for normal pixels
        assert cmap[1][1] > cmap[1][0]  # G > R
        assert cmap[1][1] > cmap[1][2]  # G > B

    def test_red_for_defective(self):
        cmap = create_wafer_colormap()
        # Red channel should be highest for defective pixels
        assert cmap[2][0] > cmap[2][1]  # R > G
        assert cmap[2][0] > cmap[2][2]  # R > B


class TestWaferMapToBase64Png:
    """Tests for wafer_map_to_base64_png function."""

    def test_returns_valid_base64_string(self):
        wafer = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.uint8)
        result = wafer_map_to_base64_png(wafer, size=32)
        # Should be a valid base64 string
        decoded = base64.b64decode(result)
        # PNG files start with specific magic bytes
        assert decoded[:4] == b"\x89PNG"

    def test_different_sizes_produce_output(self):
        wafer = np.ones((10, 10), dtype=np.uint8)
        for size in [32, 64, 128]:
            result = wafer_map_to_base64_png(wafer, size=size)
            assert len(result) > 0

    def test_rejects_non_2d_input(self):
        wafer_3d = np.zeros((3, 3, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="must be 2D"):
            wafer_map_to_base64_png(wafer_3d)

    def test_handles_all_zeros(self):
        wafer = np.zeros((8, 8), dtype=np.uint8)
        result = wafer_map_to_base64_png(wafer, size=32)
        assert len(result) > 0

    def test_handles_all_twos(self):
        wafer = np.full((8, 8), 2, dtype=np.uint8)
        result = wafer_map_to_base64_png(wafer, size=32)
        assert len(result) > 0


class TestCreateHoverTemplate:
    """Tests for create_hover_template function."""

    def test_includes_class_label(self):
        html = create_hover_template("Center", 42)
        assert "Center" in html
        assert "42" in html

    def test_includes_lot_number(self):
        html = create_hover_template("Donut", 10, lot_number="LOT_A123")
        assert "LOT_A123" in html

    def test_includes_distance(self):
        html = create_hover_template("Scratch", 5, distance_to_centroid=0.1234)
        assert "0.1234" in html

    def test_includes_thumbnail_img_tag(self):
        fake_b64 = base64.b64encode(b"fake_png_data").decode()
        html = create_hover_template("Edge-Loc", 0, base64_png=fake_b64)
        assert "<img" in html
        assert fake_b64 in html
        assert "data:image/png;base64," in html

    def test_no_thumbnail_when_none(self):
        html = create_hover_template("Loc", 7)
        assert "<img" not in html


class TestCreateScatterWithClassColors:
    """Tests for create_scatter_with_class_colors function."""

    def test_creates_figure_with_traces(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(30)
        y = rng.standard_normal(30)
        labels = np.array(["Center"] * 10 + ["Donut"] * 10 + ["Scratch"] * 10)

        fig = create_scatter_with_class_colors(x, y, labels, title="Test Scatter")
        assert isinstance(fig, go.Figure)
        # Should have 3 traces (one per class)
        assert len(fig.data) == 3

    def test_trace_names_match_classes(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        labels = np.array(["Center", "Donut", "Scratch"])

        fig = create_scatter_with_class_colors(x, y, labels)
        trace_names = {t.name for t in fig.data}
        assert trace_names == {"Center", "Donut", "Scratch"}

    def test_uses_webgl_by_default(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        labels = np.array(["Center", "Center"])

        fig = create_scatter_with_class_colors(x, y, labels, use_webgl=True)
        # Scattergl traces
        assert all(isinstance(t, go.Scattergl) for t in fig.data)

    def test_non_webgl_mode(self):
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        labels = np.array(["Center", "Center"])

        fig = create_scatter_with_class_colors(x, y, labels, use_webgl=False)
        assert all(isinstance(t, go.Scatter) for t in fig.data)

    def test_with_integer_labels_and_class_names(self):
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        labels = np.array([0, 1, 2])
        class_names = ["Center", "Donut", "Scratch"]

        fig = create_scatter_with_class_colors(x, y, labels, class_names=class_names)
        assert len(fig.data) == 3


class TestResizeNearest:
    """Tests for _resize_nearest helper."""

    def test_upscale(self):
        img = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        resized = _resize_nearest(img, 4, 4)
        assert resized.shape == (4, 4)

    def test_downscale(self):
        img = np.arange(16).reshape(4, 4).astype(np.uint8)
        resized = _resize_nearest(img, 2, 2)
        assert resized.shape == (2, 2)

    def test_preserves_channels(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = [255, 0, 0]
        resized = _resize_nearest(img, 8, 8)
        assert resized.shape == (8, 8, 3)
        # Top-left corner should still be red
        np.testing.assert_array_equal(resized[0, 0], [255, 0, 0])

    def test_identity_resize(self):
        img = np.arange(9).reshape(3, 3).astype(np.uint8)
        resized = _resize_nearest(img, 3, 3)
        np.testing.assert_array_equal(resized, img)
