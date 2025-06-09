from google import genai
import os
import json
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

client = genai.Client(api_key="AIzaSyBgY-QkVT9r2FEI43xWsQvkl_Z8BxZWzb0")

from pydantic import BaseModel
from enum import Enum

class NodeLabel(Enum):
    planning = "planning"
    fact = "fact"
    reasoning = "reasoning"
    restatement = "restatement"
    assumption = "assumption"
    example = "example"
    reflection = "reflection"
    conclusion = "conclusion"

class NodeResponse(BaseModel):
    text: str
    label: NodeLabel

class EdgeLabel(Enum):
    REASON_PREMISE_CONCLUSION = "reason:premise-conclusion"
    REASON_PLAN_STEP = "reason:plan-step"
    REASON_CONCEPT_EXAMPLE = "reason:concept-example"
    REASON_FACT_DETAIL = "reason:fact-detail"
    REASON_STMT_RESTATEMENT = "reason:stmt-restatement"
    REASON_STMT_CORRECTION = "reason:stmt-correction"
    PLAN_FRONTIER_PLAN = "plan:frontier-plan"
    PLAN_FRONTIER_VERIFY = "plan:frontier-verify"
    PLAN_PLAN_SUBPLAN = "plan:plan-subplan"
    PLAN_PLAN_NEXTPLAN = "plan:plan-nextplan"
    PLAN_PLAN_ALTERNATIVE = "plan:plan-alternative"
    EVALUATE_SUPPORT = "evaluate:support"
    EVALUATE_REFUTE = "evaluate:refute"
    EVALUATE_UNCERTAINTY = "evaluate:uncertainty"

class EdgeResponse(BaseModel):
    from_node_id: str
    label: EdgeLabel

def format_node_prompt(datum, example=False):
    if example:
        return f"Raw text:\n{datum['raw_text']}\n" \
               f"Output:\n{json.dumps(datum['nodes'], indent=2, ensure_ascii=False)}\n"
    else:
        return f"Raw text:\n{datum['raw_text']}\n" \
               f"Output:\n"

def format_edge_prompt(datum, example=False):
    if example:
        return f"Previous nodes:\n{json.dumps(datum['prev_steps'], indent=2, ensure_ascii=False)}\n" \
               f"Current node:\n{json.dumps(datum['current_step'], indent=2, ensure_ascii=False)}\n" \
               f"Output:\n{json.dumps(datum['edges'], indent=2, ensure_ascii=False)}\n"
    else:
        return f"Previous nodes:\n{json.dumps(datum['prev_steps'], indent=2, ensure_ascii=False)}\n" \
               f"Current node:\n{json.dumps(datum['current_node'], indent=2, ensure_ascii=False)}\n" \
               f"Output:\n"

in_token = 0
out_token = 0
price = 0
def call_llm(prompt: str, schema=None): # Gemini API
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": 0,
        } if schema is not None else {"temperature": 0}
    )
    global in_token, out_token, price
    in_token += response.usage_metadata.prompt_token_count
    out_token += response.usage_metadata.candidates_token_count
    price += response.usage_metadata.prompt_token_count * 0.15/1000000 + response.usage_metadata.candidates_token_count * 0.6/1000000

    if schema is not None:
        return json.loads(response.text)
    else:
        return response.text

def main_predict(): # Only predict spans, since edge quality is very low
    with open("parser/node_prompt.txt", "r") as f:
        NODE_PROMPT = f.read()
    with open("parser/edge_prompt.txt", "r") as f:
        EDGE_PROMPT = f.read()
    with open("parser/node_fewshot_examples.json", "r") as f:
        NODE_EXAMPLES = json.load(f)
    with open("parser/edge_fewshot_examples.json", "r") as f:
        EDGE_EXAMPLES = json.load(f)

    # Load the data
    print("<Load data...>")
    data = []
    for file in os.listdir("scp116k_data"):
        if file.endswith(".json"):
            with open(os.path.join("scp116k_data", file), "r") as f:
                datum = json.load(f)
                # if len(datum["nodes"]) > len(datum["edges"]):
                #     continue
                data.append(datum)
    print(f"Loaded {len(data)} data samples.")

    # test
    data = data[:1]
    for datum in data:
        print(f"Processing document: {datum['doc_id']}")
        contents = datum["raw_text"]["response"]
        response = call_llm(
            NODE_PROMPT
                .replace("<<example1>>", format_node_prompt(NODE_EXAMPLES[0], example=True))
                .replace("<<example2>>", format_node_prompt(NODE_EXAMPLES[1], example=True))
                .replace("<<example3>>", format_node_prompt(NODE_EXAMPLES[2], example=True))
                .replace("<<input>>", format_node_prompt({"raw_text": contents}, example=False)),
            schema=list[NodeResponse]
        )

        # Reorganize nodes
        nodes = datum["nodes"]
        # 1. add "question" nodes
        last_idx = 0
        sentences = sent_tokenize(datum["raw_text"]["question"])
        sentence_indices = []
        for sent in sentences:
            idx = datum["raw_text"]["question"].index(sent, last_idx)
            last_idx = idx + len(sent)
            sentence_indices.append(idx)
        sentence_indices.append(len(datum["raw_text"]["question"]))
        
        assert len(sentence_indices) == len(sentences) + 1, "Sentence indices do not match the number of sentences."

        for i in range(len(sentences)):
            nodes.append({
                "id": f"ctx{i}",
                "annotation": False,
                "start": sentence_indices[i],
                "end": sentence_indices[i + 1],
                "label": "context",
                "text": datum["raw_text"]["question"][sentence_indices[i]:sentence_indices[i + 1]],
                "source": "question"
            })

        # 2. add "response" nodes
        last_idx = 0
        sentence_indices = []
        for sent in response: # concatenating the "text" field should be the original response text
            idx = datum["raw_text"]["response"].index(sent["text"], last_idx)
            last_idx = idx + len(sent["text"])
            sentence_indices.append(idx)
        sentence_indices[0] = 0
        sentence_indices.append(len(datum["raw_text"]["response"]))

        final_conclusion = None
        for i in range(len(response)):
            if response[i]["label"] == "conclusion":
                final_conclusion = i
        # if final_conclusion is None:                
        #     assert False, "No conclusion found in the response."
        for i in range(len(response)):
            if response[i]["label"] == "conclusion":
                if i != final_conclusion:
                    response[i]["label"] = "reasoning"

        # assertion checks
        assert sentence_indices[0] == 0, "First sentence index should be 0."
        assert len(sentence_indices) == len(response) + 1, "Sentence indices do not match the number of sentences."
        # assert sum([int(x["label"] == "conclusion") for x in response]) == 1, f"There should be exactly one conclusion in the response but found {sum([int(x['label'] == 'conclusion') for x in response])}"

        # Append to the final nodes
        for i in range(len(response)):
            nodes.append({
                "id": f"resp{i}",
                "annotation": False,
                "start": sentence_indices[i],
                "end": sentence_indices[i + 1],
                "label": response[i]["label"],
                "text": datum["raw_text"]["response"][sentence_indices[i]:sentence_indices[i + 1]],
                "source": "response"
            })

        print(json.dumps(nodes, indent=4))

        # For each node that are from "source: response", annotate edges
        edges = datum["edges"]
        for i in range(len(nodes)):
            node = nodes[i]
            if node["source"] != "response":
                continue

            # Call the LLM to get edges
            response = call_llm(
                EDGE_PROMPT
                    .replace("<<example1>>", format_edge_prompt(EDGE_EXAMPLES[node["label"]][0], example=True))
                    .replace("<<example2>>", format_edge_prompt(EDGE_EXAMPLES[node["label"]][1], example=True))
                    .replace("<<example3>>", format_edge_prompt(EDGE_EXAMPLES[node["label"]][2], example=True))
                    .replace("<<input>>", format_edge_prompt({
                        "prev_steps": [
                            {
                                "id": n["id"],
                                "text": n["text"],
                                "label": n["label"]
                            } for n in nodes[:i]
                        ],
                        "current_node": node
                    })),
                schema=list[EdgeResponse]
            )

            for edge in response:
                edges.append({
                    "id": f"e{len(edges)}",
                    "from_node_id": edge["from_node_id"],
                    "to_node_id": node["id"],
                    "label": edge["label"]
                })
                print(edges[-1])

        with open(f"scp116k_data_preprocessed/{datum['doc_id']}.json", "w") as f:
            json.dump(datum, f, indent=4)
    print(f"Total input tokens: {in_token}, output tokens: {out_token}, price: ${price:.6f}")

if __name__ == "__main__":
    main_predict()
    # response = client.models.generate_content(
    #     model="gemini-2.0-flash",
    #     contents=SEGMENTATION_PROMPT,
    #     examples=data[:2],
    # )
    # print(response.text)
    # with open("scp116k_data/gemini_labeler.json", "w") as f:
    #     json.dump(response.text, f, indent=4)