import json
import os
import laya


# ============================================================
# 配置
# ============================================================

INPUT_FILE = "./dataset/candidates.json"
MODEL_PATH = "./model"

OUTPUT_DIR = "./jev_concept_data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "jev_concept_results.json"
)

# 第一次测试只跑 1 条
# 确认无误后改成 False 跑全部专利
TEST_ONLY_ONE = False


# ============================================================
# 创建输出目录
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 读取专利数据
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    patents = json.load(f)

print(f"Total patents: {len(patents)}")


# ============================================================
# 加载 Laya 模型
# ============================================================

print("Loading Laya model...")

agent = laya.load(MODEL_PATH)

print("Model loaded.")


# ============================================================
# 测试数量
# ============================================================

if TEST_ONLY_ONE:
    patents_to_process = patents[:1]
else:
    patents_to_process = patents


# ============================================================
# 结果
# ============================================================

results = []


# ============================================================
# 开始处理
# ============================================================

for i, patent in enumerate(patents_to_process):

    patent_id = patent["patent_id"]

    title = patent.get(
        "title",
        ""
    )

    description = patent.get(
        "description",
        ""
    )

    candidates = patent.get(
        "candidate_concepts",
        []
    )

    print(
        f"\n[{i + 1}/{len(patents_to_process)}] "
        f"{patent_id}"
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state = {
        "title": title,
        "description": description
    }

    # --------------------------------------------------------
    # 构造 Choice
    # --------------------------------------------------------

    criteria = {}

    for concept in candidates:

        concept_id = concept["id"]

        concept_name = concept.get(
            "display_name",
            ""
        )

        concept_description = concept.get(
            "description"
        )

        if concept_description:

            criteria[concept_id] = (
                f"{concept_name}: "
                f"{concept_description}"
            )

        else:

            criteria[concept_id] = concept_name

    questions = {
        "concept": {
            "type": "choice",
            "instructions": (
                "Which concept best represents "
                "the main research topic and "
                "technical contribution of this patent?"
            ),
            "criteria": criteria
        }
    }

    # --------------------------------------------------------
    # Laya 判断
    # --------------------------------------------------------

    try:

        result = agent.predict(
            state,
            questions
        )

        answer = result["answers"]["concept"]

        # Laya 选择的 Concept ID
        selected_id = answer["choice"]

        # Laya confidence
        confidence = answer["confidence"]

    except Exception as e:

        print(
            f"[ERROR] {patent_id}: {e}"
        )

        continue

    # --------------------------------------------------------
    # 根据 selected_id 找到 Concept 的文字名称
    # --------------------------------------------------------

    selected_concept = None

    for concept in candidates:

        if concept["id"] == selected_id:

            selected_concept = concept.get(
                "display_name",
                ""
            )

            break

    # --------------------------------------------------------
    # 没找到 Concept
    # --------------------------------------------------------

    if selected_concept is None:

        print(
            f"[ERROR] Cannot find selected "
            f"concept: {selected_id}"
        )

        continue

    # --------------------------------------------------------
    # 最终结果
    #
    # 只保存：
    # patent_id
    # title
    # concept
    # confidence
    # --------------------------------------------------------

    item = {
        "patent_id": patent_id,
        "title": title,
        "concept": selected_concept,
        "confidence": confidence
    }

    results.append(item)

    # --------------------------------------------------------
    # 终端输出
    # --------------------------------------------------------

    print(
        f"Title: {title}"
    )

    print(
        f"Concept: {selected_concept}"
    )

    print(
        f"Confidence: {confidence}"
    )


# ============================================================
# 保存最终结果
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 完成
# ============================================================

print()
print("=" * 60)
print("Finished.")
print("=" * 60)

print(
    f"Processed: {len(results)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)
