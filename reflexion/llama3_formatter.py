# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from logging import getLogger
from typing import (
    List,
    Literal,
    Sequence,
    TypedDict,
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


logger = getLogger(__name__)


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

class ChatFormat:
    def __init__(self):
        self.sh_id = "<|start_header_id|>"
        self.eh_id = "<|end_header_id|>"
        self.e_id = "<|eot_id|>"
        self.bot_id = "<|begin_of_text|>"

    def format_header(self, message: Message) -> str:
        return f"{self.sh_id}{message['role']}{self.eh_id}\n\n"

    def format_message(self, message: Message) -> str:
        return f"{self.format_header(message)}{message["content"].strip()}{self.e_id}"

    def format_dialog_prompt(self, dialog: Dialog) -> str:
        messages = ''.join(self.format_message(message) for message in dialog)
        assistant_header = self.format_header({"role": "assistant", "content": ""})
        return f"{self.bot_id}{messages}{assistant_header}"
