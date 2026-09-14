import json
from pathlib import Path

from convert_mineru_content_list import convert_mineru_content_list_v2
from structure_aware import structure_aware

if __name__ == "__main__":
    pdf = (
        Path(__file__).resolve().parent
        / "mineru-output/浦发上海浦东发展银行西安分行个金客户经理考核办法/hybrid_auto/浦发上海浦东发展银行西安分行个金客户经理考核办法_content_list_v2.json"
    )
    payload = json.loads(pdf.read_text(encoding="utf-8"))
    converted = convert_mineru_content_list_v2(payload)
    sections = structure_aware(converted)
    print(json.dumps(sections, ensure_ascii=False))
