## Multimodal RAG

1. Collect video from youtube using miltimodal_rag_preparation_functions.collect_video_youtube
It collects the diarised video transcription using https://replicate.com/thomasmol/whisper-diarization.
Alternatively, use https://replicate.com/thomasmol/whisper-diarization directly but you'll need a link produced by collect_video_youtube anyways, to pass it to the replicate model.

2. Clean json with diarised transcription got from p.1 using miltimodal_rag_preparation_functions.extract_only_word_ts_from_transcript.

3. Extract full transcript from json with diarised transcription got from p.1 using miltimodal_rag_preparation_functions.extract_full_text_from_diarised_transcript.

4. Feed the full text to Claude or other model using the following prompt:
------------
Here's a transcript of the video:

```text
[transcript]
```

I need you to break it down into meaningful parts and name each.

Output the result in a valid json format with the following structure:

```json

[

{"name": "[part 1 name]", "start": "[first 7 words of the part]", "finish": "[last 7 words of the part]"(parts can intersect for not more than 20%) }

]

```
------------

4. Use miltimodal_rag_preparation_functions.process_segments to get the final json of segments with timestamps of the start and end phrases and full segments texts.

5. See how to extract relevant screenshots from the video and create their description + test questions in the conversation with Claude: claude_conversation_segmenting_transcript_and finding_screenshots.md.


## Recursive RAG

1. Generate agreements in Claude.

2. 
