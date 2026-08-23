import torch

from bevformer.models.transformer.positional_encoding import LearnedBEVPositionalEncoding


def test_output_shape():
    bev_h, bev_w, embed_dims = 4, 5, 8
    pos_enc = LearnedBEVPositionalEncoding(bev_h, bev_w, embed_dims)
    output = pos_enc(batch_size=3)
    assert output.shape == (3, bev_h * bev_w, embed_dims)


def test_different_cells_get_different_encodings():
    bev_h, bev_w, embed_dims = 4, 4, 8
    pos_enc = LearnedBEVPositionalEncoding(bev_h, bev_w, embed_dims)
    output = pos_enc(batch_size=1)
    assert not torch.allclose(output[0, 0], output[0, 1])


def test_same_batch_index_shares_identical_encoding():
    bev_h, bev_w, embed_dims = 3, 3, 8
    pos_enc = LearnedBEVPositionalEncoding(bev_h, bev_w, embed_dims)
    output = pos_enc(batch_size=2)
    torch.testing.assert_close(output[0], output[1])
