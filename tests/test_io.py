import json
import data_io

def test_json_roundtrip(tmp_path):
    dst = tmp_path / 'value.json'
    data_io.atomic_json({'x':[1,2], 'status':'ok'},dst)
    assert json.loads(dst.read_text()) == {'x':[1,2], 'status':'ok'}
    assert len(data_io.sha256_file(dst)) == 64
