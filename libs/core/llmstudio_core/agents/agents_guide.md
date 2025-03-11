## LLMstudio Agents Documentation

This is a brief overveiw of the main functions and data types you need to be aware of to interact with LLMStudio agents, to see examples of the usage of this wrapper with openai and bedrock agents see [`weather_example_bedrock.py`](https://github.com/TensorOpsAI/LLMstudio/blob/678e7b6a1c68a08ebcc7409722954d41bd02d0ac/examples/weather_example_bedrock.py) and [`weather_example_openai.py`](https://github.com/TensorOpsAI/LLMstudio/blob/678e7b6a1c68a08ebcc7409722954d41bd02d0ac/examples/weather_example_openai.py) in the examples folder

### Functions

#### `create_agent`:


Function to create a new agent / assistant, takes a single `CreateAgentRequest` object as input that defines the fields of the newly created agent.

Parameters:
- request: `CreateAgentRequest`

Returns:
- `AgentBase` - agent object to be used when running requests


#### `run_agent`:

Creates a run request for the agent in question, this function is *non-blocking*, meaning it won't return a final result, simple a reference to a run to be later polled / retrieved.

Parameters:
- request: `RunAgentRequest`

Returns:
- `RunBase` - run object that can be used to retrieve the results

#### `retrieve_result`:

Retireves the result based on a provided reference to a previously started run, waits until the run is done in a synchronous way. This should

Parameters:
- request: `RunBase`

Returns:
- `ResultBase` - result of the run

#### `submit_tool_outputs`:

Parameters:
- request: `RunBase`

Returns:
- `ResultBase` - result of the run


### Data Types

#### `CreateAgentRequest`

`CreateAgentRequest` is a data type used to define the parameters required to create a new agent.

Attributes:
- `model` (str): The foundational model identifier for the agent.
- `instructions` (Optional[str]): System prompt for the agent
- `description` (Optional[str]): Agent description
- `tools` (Optional[list[`Tool`]]): Tools the agent can use, specified in openai format
- `name` (Optional[str]): Name for the agent.
- `tool_resources` (Optional[`ToolResources`]): resources for code interpreter and vector search
- `agent_resource_role_arn` (Optional[str]) **REQUIRED FOR BEDROCK**: ARN for resource access role.
- `agent_alias` (Optional[str]) **REQUIRED FOR BEDROCK**: Alias for the agent

#### `RunAgentRequest`:

`RunAgentRequest` is a data type used to define the parameters required to run an existing agent.

Attributes:
- `agent_id` (str): The unique identifier of the agent to be run.
- `alias_id` (Optional[str]) **REQUIRED FOR BEDROCK** : The alias identifier of the agent.
- `thread_id` (Optional[str]): An optional identifier for the thread or conversation context. If one is not provided, a new thread will be create.
- `messages` (Optional[List[`Message`]]): A list of messages representing the conversation history. Should only be used when using the RunAgentRequest as an input to the `run_agent` function
- `tool_outputs` (Optional[List[`ToolOutput`]]): A list of outputs from tools that the agent can utilize. This should only be used when using the RunAgentRequest as an input to the `submit_tool_outputs` function
- `run_id` (Optional[str]): An optional identifier to continue a previous run. his should only be used when using the RunAgentRequest as an input to the `submit_tool_outputs` function

Functions:

- `from_agent`: An auxiliary to create a `RunAgentRequest` instance from an `AgentBase` object.

    Parameters:
    - `agent` (`AgentBase`): The agent from which to create the request.
    - `thread_id` (Optional[str]): An optional identifier for the thread or conversation context.
    - `messages` (Optional[List[`Message`]]): A list of messages representing the conversation history.
    - `tool_outputs` (Optional[List[`ToolOutput`]]): A list of outputs from tools that the agent can utilize.
    - `run_id` (Optional[str]): An optional identifier to continue a previous run.

    Returns:
    - `RunAgentRequest`: A new instance of `RunAgentRequest` initialized with the provided parameters.


#### `Tool`

`Tool` is a data type used to specify tools that an agent can utilize.

Attributes:
- `type` (str): The type identifier of the tool.
- `function` (Optional[`Function`]): The function definition associated with the tool, specified in OpenAI format.


#### `Function`

`Function` is a data type used to define a function associated with a tool, specified in OpenAI function format.

Attributes:
- `name` (str): The name of the function.
- `description` (str): A brief description of the function's purpose.
- `parameters` (`Parameters`): The parameters accepted by the function.


#### `Parameters`

`Parameters` is a data type used to define the parameters accepted by a function in OpenAI function format.

Attributes:
- `type` (str): The type of the parameters object (typically set to "object").
- `properties` (Dict): A dictionary specifying the properties (parameters) that the function accepts.
- `required` (List[str]): A list of parameter names that are required.


#### `ToolResources`

`ToolResources` is a data type that contains resources required by certain tools used by the agent.

Attributes:
- `file_ids` (Optional[List[str]]): A list of file identifiers for use with the `code_interpreter` tool.
- `vector_store_ids` (Optional[List[str]]): A list of vector store identifiers for use with the `file_search` tool.


#### `ToolOutput`

`ToolOutput` is a data type that represents the output from a tool after its execution by the agent.

Attributes:
- `tool_call_id` (Optional[str]): The unique identifier of the tool call instance.
- `output` (Optional[str]): The result or output produced by the tool.
- `action_group` (Optional[str]): The action group or category associated with the tool call.
- `function_name` (Optional[str]): The name of the function associated with the executed tool.

Functions:
- `from_tool_call`: Class method that creates a `ToolOutput` instance from a given `ToolCall` and its output.

    Parameters:

    - `tool_call` (`ToolCall`): The `ToolCall` instance representing the tool call.
    - `tool_output` (`str`): The output produced by the tool.

    Returns:

    - `ToolOutput`: An instance of `ToolOutput` populated with data from the `ToolCall` and `tool_output`.



#### `RunBase`

`RunBase` represents the essential information required to initiate or continue an agent's run within a conversation thread.

Attributes:
- `thread_id` (str): The unique identifier of the conversation thread.
- `agent_id` (Optional[str]): The unique identifier of the agent handling the thread. Defaults to `None`.

#### `ResultBase`

`ResultBase` is a data type that encapsulates the result produced by the agent after processing a request.

Attributes:
- `thread_id` (str): The unique identifier of the conversation thread.
- `messages` (List[`Message`]): A list of `Message` instances representing the conversation history, including both user and agent messages.
- `run_id` (Optional[str]): The unique identifier of the specific run or interaction. Defaults to `None`.
- `usage` (Optional[dict]): A dictionary containing usage metrics, such as token counts and processing time. Defaults to `None`.
- `run_status` (Optional[str]): The current status of the run (e.g., `'completed'`, `'in-progress'`, `'failed'`). Defaults to `None`.
- `required_action` (Optional[`RequiredAction`]): An optional field indicating any required user action to proceed further. Defaults to `None`.


#### `RequiredAction`

`RequiredAction` is a data type representing an action required from the user to proceed with the interaction, typically involving the submission of tool outputs.

Attributes:
- `submit_tool_outputs` (List[`ToolCall`]): A list of `ToolCall` instances that the user needs to submit.
- `type` (Literal["submit_tool_outputs"]): A literal string indicating the type of required action. Defaults to `"submit_tool_outputs"`.

#### `Message`

`Message` represents an individual message within a conversation thread between a user and an assistant.

Attributes:
- `id` (Optional[str]): The unique identifier of the message. Defaults to `None`.
- `object` (Optional[str]): The type of the object, defaulting to `"thread.message"`.
- `created_at` (Optional[int]): A timestamp indicating when the message was created. Defaults to `None`.
- `thread_id` (Optional[str]): The unique identifier of the conversation thread. Defaults to `None`.
- `role` (Optional[Literal["user", "assistant"]]): The role of the message sender, either `"user"` or `"assistant"`. Defaults to `None`.
- `content` (Optional[Union[str, List[Union[`ImageFileContent`, `TextContent`, `RefusalContent`, `ImageUrlContent`]]]]): The content of the message, which can be a string or a list of content objects. Defaults to an empty list.
- `assistant_id` (Optional[str]): The unique identifier of the assistant that generated the message. Defaults to `None`.
- `run_id` (Optional[str]): The unique identifier of the specific run or interaction. Defaults to `None`.
- `attachments` (List[`Attachment`]): A list of attachment objects associated with the message. Defaults to an empty list.
- `metadata` (Optional[dict]): A dictionary containing additional metadata for the message. Defaults to an empty dictionary.
- `required_action` (Optional[`RequiredAction`]): An optional field indicating any required action from the user to proceed further. Defaults to `None`.
