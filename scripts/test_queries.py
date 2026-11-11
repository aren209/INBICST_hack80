from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_search import HybridRetriever

SAMPLE_QUERIES = [
    "Каков порядок открытия финансового лицевого счета (ФЛС) в 1С:Документооборот и чем он отличается от старого порядка?",
    "Какой размер суточных при командировках по России для работников и обучающихся МФТИ?",
    "Как формируется 15-значный номер ФЛС и какие коды КФО используются?",
    "Как подать заявку на бронирование конференц-зала МФТИ и за сколько дней?",
    "Какие правила доступа и эксплуатации центра обработки данных (ЦОД) МФТИ?",
    "Каков порядок оформления корректировки сметы доходов и расходов проекта?",
    "Какие правила подачи заявки на поселение сотрудников в общежитие?",
    "Как приготовить пасту карбонара в домашних условиях?",
]


def main() -> None:
    retriever = HybridRetriever()
    for q in SAMPLE_QUERIES:
        print("=" * 90)
        print(f"ВОПРОС: {q}")
        res = retriever.search(q, top_k=4)
        print(
            f"max_dense_score={res.max_dense_score:.4f} | lexical_hits={res.lexical_hits} | "
            f"abstain={res.abstain_recommended}"
        )
        if res.abstain_recommended:
            print(f"  [ОТКАЗ ОТ ОТВЕТА]: {res.abstain_reason}")
            continue
        if res.version_alerts:
            for va in res.version_alerts:
                print(f"  [ВЕРСИОННАЯ КОЛЛИЗИЯ]: {va}")
        if res.quality_alerts:
            for qa in res.quality_alerts:
                print(f"  [ПРЕДУПРЕЖДЕНИЕ ПО СКАНУ]: {qa}")
        for i, h in enumerate(res.hits, start=1):
            snippet = h.raw_text.replace("\n", " ")[:180]
            print(
                f"  #{i} [RRF={h.rrf_score:.3f} | Dense={h.dense_score:.3f}] "
                f"{h.doc_title} ({h.status}) -> {h.unit_title} -> Раздел {h.section_no} -> Пункт '{h.clause_no}' (стр. {h.page_from}):\n"
                f"     {snippet}..."
            )


if __name__ == "__main__":
    main()
