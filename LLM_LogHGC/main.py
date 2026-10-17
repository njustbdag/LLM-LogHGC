import random
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import re
from llm import FlexibleLogTemplateExtractor
from temple_cache import TemplateTree
import ast
import MAHGCN

def get_input_log(related, subhypergraph, id2word):
    sub_logs = [[id2word[idx] for idx in seq] for seq in subhypergraph]
    _structure = list(map(list, set(map(tuple, related.values()))))
    sub_structure = [[id2word[_idx] for _idx in _seq] for _seq in _structure]
    structure_id_map = {
        tuple(structure): idx for idx, structure in enumerate(_structure)
    }
    sub_structure_id = []
    sub_structure_id.append(-1)
    for shared_nodes in related.values():
        sub_structure_id.append(structure_id_map[tuple(shared_nodes)])
    return sub_logs, sub_structure, sub_structure_id


def get_related_hyperedges(
    edge_id,
    sentence_word_id_list,
    inverted_index,
    max_per_type=200,
    random_sample=True,
    id2word=None,
):
    target_edge = sentence_word_id_list[edge_id]
    neighbors = {}
    for token in set(target_edge):
        # 不通过变量占位符检索邻居，也不将其计入共享结构。
        if id2word is not None and id2word[token].lower() == "<x>":
            continue
        for neigh_id in inverted_index.get(token, []):
            if neigh_id == edge_id:
                continue
            if neigh_id not in neighbors:
                neighbors[neigh_id] = set()
            neighbors[neigh_id].add(token)
    intersection_groups = defaultdict(list)
    for neigh_id, intersection in neighbors.items():
        intersection_key = tuple(sorted(intersection))
        intersection_groups[intersection_key].append(neigh_id)
    selected_neighbors = {}
    subhypergraph = [sentence_word_id_list[edge_id]]
    for intersection_key, neigh_list in intersection_groups.items():
        if len(neigh_list) > max_per_type:
            if random_sample:
                neigh_list = random.sample(neigh_list, max_per_type)
            else:
                neigh_list = neigh_list[:max_per_type]
        for neigh_id in neigh_list:
            selected_neighbors[neigh_id] = list(intersection_key)
            subhypergraph.append(sentence_word_id_list[neigh_id])
    return selected_neighbors, subhypergraph


def load_or_build_inverted_index(
    sentence_word_id_list, index_file="inverted_index.npz"
):
    if os.path.exists(index_file):
        print(f"Loading inverted index from {index_file}")
        data = np.load(index_file, allow_pickle=True)
        inverted_index = data["inverted_index"].item()
        print(f"Loaded inverted index, " f"tokens={len(inverted_index)}")
        return inverted_index
    print("Building inverted index...")
    inverted_index = defaultdict(list)
    for edge_id, edge in enumerate(sentence_word_id_list):
        for token in set(edge):
            inverted_index[token].append(edge_id)
        if edge_id % 1000000 == 0:
            print(f"processed {edge_id}")

    inverted_index = {
        token: np.asarray(edge_ids, dtype=np.int32)
        for token, edge_ids in inverted_index.items()
    }
    np.savez_compressed(index_file, inverted_index=inverted_index)
    print(f"Saved inverted index to {index_file}, " f"tokens={len(inverted_index)}")
    return inverted_index


def load_or_build_data(csv_file, npz_file="log_data.npz"):
    if os.path.exists(npz_file):
        print(f"load local data {npz_file} ...")
        data = np.load(npz_file, allow_pickle=True)
        sentence_word_id_list = data["sentence_word_id_list"]
        word2id = data["word2id"].item()
    else:
        print(f"building log data {csv_file} ...")
        word2id = {}
        next_id = 0
        sentence_word_id_list = []

        for s in pd.read_csv(csv_file, usecols=["Content"])["Content"]:
            s = s.replace("|", " ")
            s = s.replace("[", "").replace("]", "")
            s = re.sub(r"=(?! )", "{Equal} ", s)
            IP_REGEX = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
            UUID_REGEX = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
            TIME_REGEX = (
                r"\b(?:(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?|[0-5]?\d:[0-5]\d)\b"
            )
            DATE_REGEX = r"\b\d{4}-\d{2}-\d{2}\b"
            PATH_REGEX = r"\/(?:[A-Za-z0-9_.-]+\/)+[A-Za-z0-9_.-]+\/?"
            HOSTPORT_REGEX = r"\b(?:((?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}|localhost|\d{1,3}(?:\.\d{1,3}){3}):\d{1,5})\b"
            UNIT_ABBR = r"\b[<>]?\d+(\.\d+)?\s*(KB|MB|GB|TB|sec|min|hour|bytes)\b"
            s = re.sub(HOSTPORT_REGEX, "<X>", s)
            s = re.sub(IP_REGEX, "<X>", s)
            s = re.sub(UUID_REGEX, "<X>", s)
            s = re.sub(TIME_REGEX, "<X>", s)
            s = re.sub(PATH_REGEX, "<X>", s)
            s = re.sub(DATE_REGEX, "<X>", s)
            s = re.sub(UNIT_ABBR, "<X>", s)
            word_ids = []
            for word in s.split():
                if word not in word2id:
                    word2id[word] = next_id
                    next_id += 1
                word_ids.append(word2id[word])
            sentence_word_id_list.append(word_ids)

        np.savez_compressed(
            npz_file,
            sentence_word_id_list=np.array(sentence_word_id_list, dtype=object),
            word2id=word2id,
        )
    id2word = {v: k for k, v in word2id.items()}
    print("Log loaded ...")
    return sentence_word_id_list, id2word


def sort_length(reader_file):
    reader = pd.read_csv(reader_file)
    data_list = reader["Content"].tolist()
    return data_list


def extract_list_from_text(text: str):
    start = -1
    depth = 0
    last = None
    for i, ch in enumerate(text):
        if ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start != -1:
                last = text[start : i + 1]
    if last is None:
        return None
    try:

        lst = ast.literal_eval(last)
        return [str(x) for x in lst]
    except (ValueError, SyntaxError):
        repaired = re.sub(
            r"""(?P<prefix>[\[,]\s*)(?P<quote>['"])<X>(?P=quote)(?P<suffix>[).:;!?]+)(?P=quote)(?=\s*[,\]])""",
            lambda m: (m["prefix"] + m["quote"] + "<X>" + m["suffix"] + m["quote"]),
            last,
        )
        if repaired == last:
            return None
        try:
            lst = ast.literal_eval(repaired)
            return [str(x) for x in lst]
        except (ValueError, SyntaxError):
            return None


def validate_content(raw_log, parsing_template):
    if len(raw_log) != len(parsing_template):
        return False
    for log_item, template_item in zip(raw_log, parsing_template):
        if template_item == "<X>":
            continue
        if log_item != template_item:
            return False
    return True


def validate_structure(parsing_template, v_structure):
    template_core = [x for x in parsing_template if x != "<X>"]
    template_set = set(template_core)
    return any(template_set <= set(v) for v in v_structure)


def llm_parsing(extractor, log_messages, custom_prompt):
    log2 = log_messages
    custom_prompt2 = custom_prompt
    llm_output = extractor.extract_template(log2, custom_prompt2)
    return llm_output


if __name__ == "__main__":
    file_path_list = [
        "Apache",
        "BGL",
        "Hadoop",
        "HDFS",
        "HealthApp",
        "HPC",
        "Linux",
        "Mac",
        "OpenSSH",
        "OpenStack",
        "Proxifier",
        "Spark",
        "Thunderbird",
        "Zookeeper",
    ]
    data_name = file_path_list[0]
    file_path = f"datasets/{data_name}/{data_name}_full.log_structured.csv"
    npz_file = f"datasets/{data_name}/{data_name}_log_data.npz"
    npz_inverted_file = f"datasets/{data_name}/{data_name}_inverted_index.npz"
    sentence_word_id_list, id2word = load_or_build_data(file_path, npz_file)
    inverted_index = load_or_build_inverted_index(
        sentence_word_id_list, npz_inverted_file
    )
    log_messages_list = sort_length(file_path)
    if len(log_messages_list) != len(sentence_word_id_list):
        raise ValueError("Input CSV and cached log data have different row counts.")

    # template cache
    template_pool = []
    tree = TemplateTree()
    for tem in template_pool:
        tree.insert(tem)

    # do log parsing
    results = {}
    _logs = []
    result_status = {}
    for parsing_round in range(2):
        if parsing_round == 0:
            pending_logs = enumerate(log_messages_list)
        else:
            pending_logs = _logs
            _logs = []
            print(f"Retrying {len(pending_logs)} failed logs...")
        for log_index, parsing_message in pending_logs:
            parsing_log_list = [id2word[i] for i in sentence_word_id_list[log_index]]
            matched = tree.match(parsing_log_list)
            if matched is not None:
                results[log_index] = list(matched)
                result_status[log_index] = (
                    "cache" if parsing_round == 0 else "retry_cache"
                )
                continue
            else:
                print(f"Not match：{parsing_message}")
                related, subhypergraph = get_related_hyperedges(
                    edge_id=log_index,
                    sentence_word_id_list=sentence_word_id_list,
                    inverted_index=inverted_index,
                    id2word=id2word,
                )

                log_sh, v_structure, clique_id = get_input_log(
                    related, subhypergraph, id2word
                )
                print("structure:", len(log_sh), v_structure)
                examples = MAHGCN.get_example(log_sh, data_name, clique_id)
                examples_text = "\n".join(str(item) for item in examples)

                # prompt
                custom_prompt = f"""
                You are a log parsing expert, skilled at identifying dynamic variables in logs.
                Task: The input consists of a target log message (represented as a list) along with several similar log messages. Use the similar logs as contextual references to identify variable tokens in the target log message and replace them with '<X>'.
                Please strictly adhere to the following rules:
                1. Please analyze each element in the list one by one, determine whether it is a variable.
                2. Do not modify any words except for replacing variables with '<X>'.
                3. Output the list of replaced log templates.
                Similar log messages (possibly sharing the same template):
                {examples_text}
                Now please parse the following target log message:
                {parsing_log_list}
                """

                extractor = FlexibleLogTemplateExtractor()
                MAX_RETRY = 3
                parsing_success = False
                for times in range(MAX_RETRY):
                    llm_output = llm_parsing(extractor, parsing_message, custom_prompt)
                    part_template2 = extract_list_from_text(llm_output)
                    if not part_template2:
                        print("LLM output error")
                        continue
                    masked_template = [
                        "<X>" if "<X>" in p else p for p in part_template2
                    ]
                    masked_template2 = [
                        "<X>" if re.fullmatch(r"-?\d+(\.\d+)?", x) else x
                        for x in masked_template
                    ]
                    if not validate_content(parsing_log_list, masked_template2):
                        print("cant match")
                        continue
                    if not validate_structure(masked_template2, v_structure):
                        print("Structure not match:", masked_template2)
                        continue

                    parsing_success = True
                    print(f"log: {parsing_message}")
                    break

                if parsing_success:
                    print("template：", masked_template2)
                    results[log_index] = masked_template2
                    result_status[log_index] = (
                        "parsed" if parsing_round == 0 else "retry_parsed"
                    )
                    if masked_template2 not in template_pool:
                        template_pool.append(masked_template2)
                        tree.insert(masked_template2)
                else:
                    _logs.append((log_index, parsing_message))
                    results[log_index] = None
                    result_status[log_index] = "failed"

    tree.show_store()
    print(f"Remaining failed logs: {len(_logs)}")
    output_data = pd.read_csv(file_path)
    if len(output_data) != len(sentence_word_id_list):
        raise ValueError("Input CSV and cached log data have different row counts.")
    output_data["PredictedTemplate"] = [
        " ".join(results[i]) if results[i] is not None else ""
        for i in range(len(output_data))
    ]
    output_data["ParseStatus"] = [result_status[i] for i in range(len(output_data))]
    template_group_ids = {}
    for template in results.values():
        if template is not None and tuple(template) not in template_group_ids:
            cached_template = tree.find_compatible(template)
            if cached_template is None or cached_template.group_id is None:
                raise ValueError(
                    f"Result template is missing from the cache: {template}"
                )
            template_group_ids[tuple(template)] = str(cached_template.group_id)
    output_data["TemplateGroupId"] = [
        template_group_ids[tuple(results[i])] if results[i] is not None else ""
        for i in range(len(output_data))
    ]
    output_file = f"datasets/{data_name}/{data_name}_parsed_results.csv"
    output_data.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Parsing results saved to {output_file}")
