please help me produce the skeleton code for a video understanding agent.

the state should be a video file name and a stack of questions:str, an arr of json of {question, answer, reason} know as QA notebook, and a json of tool call results that is of form {<tool_name>: [result_str]} 

the node should be:  setup, try_answer_with_past_QA, try_answer_with_reasoing, is_primitive_question, decide_tool_calls, execute_tool_calls, decompose_to_sub_question, answer_query

edges are:
start - setup,
set-up- try_answer_with_past_QA,
try_answer_with_past_QA - conditional {
not answered:   try_answer_with_reasoning,
answered and is not root:  try_answer_with_past_QA,
answered and is root: answer_query
}
try_answer_with_reasoning  -  conditional {
not answered:   is primitive question,
answered and is not root:  try_answer_with_past_QA,
answered and is root: answer_query
}
is primitive question - conditional {
yes: decide_tool_calls 
no: decompose_to_sub_question
}
decide_tool_calls  - execute_tool_calls
execute_tool_calls -  try_answer_with_reasoning
decompose_to_sub_question - try_answer_with_past_QA