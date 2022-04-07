import json


def longestCommonSubsequence(text1: str, text2: str) -> int:
    # 计算最长公共子序列
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def read_jsonl_file(filename):
    return [json.loads(line) for line in open(filename, "r")]


def write_jsonl_file(data, filename):
    with open(filename, "w") as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + "\n")


def generate_data_for_model(filename):
    """
        以user对话为中心，向前搜寻最近service的*3*句对话，作为候选
        将所有候选与每个summary做最长公共子序列，取得分最高的作为src-tgt pair
    """
    ori_data = read_jsonl_file(filename)
    data4model = []
    for index, one_dialogue in enumerate(ori_data):
        print("process data {}".format(index))
        dialogues, summary = one_dialogue["dialogues"], one_dialogue["summary"]
        service_dialogues = []
        user_dialogues = []
        candidates = []
        for text_id, dialogue_info in enumerate(dialogues):
            role_type = dialogue_info["role_type"]
            text = dialogue_info["text"]
            if role_type == "service":
                service_dialogues.append(text)
            if role_type == "user":
                user_dialogues.append(text)
                front_service_dialogues = service_dialogues[-3:]
                candidates.append(front_service_dialogues + [text])

        for summ in summary:
            max_lcs = 0
            fit_dialogue = None
            for candi in candidates:
                lcs = longestCommonSubsequence(summ, "".join(candi))
                if lcs > max_lcs:
                    max_lcs = lcs
                    fit_dialogue = candi.copy()
            if fit_dialogue:
                fit_dialogue[-1] = "客户：" + fit_dialogue[-1]
                for i in range(len(fit_dialogue)-1):
                    fit_dialogue[i] = "销售：" + fit_dialogue[i]
                src_txt = "#".join(fit_dialogue)
                data4model.append(
                    {
                        "src_txt": src_txt,
                        "tgt_txt": summ,
                    }
                )
    return data4model


def split_train_and_dev(data):
    import random
    random.seed(42)
    random.shuffle(data)
    length = len(data)
    train = data[:int(0.8*length)]
    dev = data[int(0.8*length):]
    return train, dev


if __name__ == "__main__":
    data = generate_data_for_model(
        "data/DTA-train/DTA-conversation-summary_train.jsonl")
    train, dev = split_train_and_dev(data)
    write_jsonl_file(train, "data/train.jsonl")
    write_jsonl_file(dev, "data/dev.jsonl")
