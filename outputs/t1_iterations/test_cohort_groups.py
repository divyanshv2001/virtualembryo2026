from cohort_groups import GROUPS

def test_proxy_groups_do_not_duplicate_atlas_cells_by_overlapping_labels():
    labels=[label for group in GROUPS.values() for label in group]
    assert len(labels)==len(set(labels))
    assert all(GROUPS.values())
    assert {'neural','somitic','extraembryonic','endoderm','cardiac'}<=set(GROUPS)
