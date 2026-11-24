import numpy as np

from prism.config import ATOMIC_LABELS, SELECTED_LABELS


def test_insert_gaps():
    from prism.data.masks import insert_gaps

    g = insert_gaps(np.arange(7514, dtype=np.float32))
    assert g.shape == (8575,) and g.dtype == np.float32
    assert g[246] == 0 and g[3273] == 3027 and g[3585] == 3028 and g[8334] == 7513
    assert (g[:246] == 0).all() and (g[3274:3585] == 0).all() and (g[6080:6344] == 0).all()
    assert (g[8335:] == 0).all()


def test_pixel_mask_from_bad(synthetic_masks):
    from prism.data.masks import load_apogee_mask, load_cno_masks, pixel_mask_from_bad, suppression_matrix

    am = load_apogee_mask(synthetic_masks)
    sup = suppression_matrix(am, load_cno_masks(synthetic_masks, 0.01), 0.01)
    assert sup.shape == (23, 8575)
    bad = np.zeros((3, 23), np.float32)
    bad[0, SELECTED_LABELS.index("MG_FE")] = 1
    bad[1, SELECTED_LABELS.index("FE_H")] = 1
    pm = np.asarray(pixel_mask_from_bad(bad, sup))
    mg_lines = am[:, 10] > 0.01
    assert (pm[0, mg_lines] == 0).all() and (pm[0, ~mg_lines] == 1).all()
    assert (pm[1] == 1).all() and (pm[2] == 1).all()  # FE_H rows never suppress; clean star


def test_line_mask_to_forbidden(synthetic_masks):
    from prism.data.masks import line_mask_to_forbidden, load_apogee_mask

    am = load_apogee_mask(synthetic_masks)
    forb = line_mask_to_forbidden(am, 0.01)
    assert not np.any((am > 0.01) & (forb > 0))
    assert (forb[:, 0] == 0).all()
    assert forb[:, 10].sum() == 8575 - (am[:, 10] > 0.01).sum()


def test_inference_element_masks(synthetic_masks):
    from prism.data.masks import element_prior_weights, inference_element_masks, load_apogee_mask

    masks, skip = inference_element_masks(load_apogee_mask(synthetic_masks), 0.01)
    assert set(masks) == set(range(9, 23)) and skip == set()
    assert masks[14].sum() == 6  # S_FE: few pixels, but no full-spectrum fallback
    pw = element_prior_weights(masks)
    assert np.isclose(pw[14], 20.0)  # 500/10 clipped to 20


def test_inference_element_masks_skips_empty():
    from prism.data.masks import inference_element_masks

    am = np.zeros((8575, 27), np.float32)
    am[:50, 10] = 0.5
    masks, skip = inference_element_masks(am, 0.01)
    assert skip == set(range(9, 23)) - {10}


def test_element_indices(synthetic_masks):
    from prism.data.masks import element_indices, load_apogee_mask

    am = load_apogee_mask(synthetic_masks)
    idx = element_indices(am, 0.01)
    assert list(idx) == ATOMIC_LABELS
    ind, w = idx["S_FE"]
    assert ind.dtype == np.int32 and w.dtype == np.float32 and len(ind) == 6
    assert np.allclose(w, am[ind, 14])
