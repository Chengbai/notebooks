from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class ImgCommentDataItem:
    source: str
    image_name: str
    comment_number: int
    comment: str
    image_id: int = None

    def to_json(self) -> Dict:
        assert self.image_id is not None
        return asdict(self)
