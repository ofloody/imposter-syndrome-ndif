from nnsight import LanguageModel, CONFIG

repo_id = "meta-llama/Llama-3.1-8B-Instruct"
peft_id = "NDIF/hackathon-imposter-syndrome-eve-llama8B"

CONFIG.API.HOST = "http://ndif-hackathon.duckdns.org:8001"

model = LanguageModel(repo_id, peft=peft_id)

with model.trace("Hello world", remote=True) as tracer:

    output = model.output.save()
