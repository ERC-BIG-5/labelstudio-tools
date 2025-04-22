import json
import os
from pathlib import Path
from typing import Annotated, Any, Optional, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field

# First, find out what metaclass BaseModel uses
pydantic_metaclass = type(BaseModel)
print(f"Pydantic BaseModel uses metaclass: {pydantic_metaclass}")


class Storage(BaseModel):
    dir: Path = None
    # We'll use a non-model field for the owner reference
    owner: Optional[BaseModel] = Field(exclude=True)

    def __repr__(self):
        return f"storage -> {self.dir}"

    def dump(self, filename=None):
        """Dump the owner model to a file in the storage directory."""
        if not self.owner:
            raise ValueError("Storage has no owner model to dump")

        # Generate filename if not provided
        if filename is None:
            if hasattr(self.owner, "name"):
                filename = f"{self.owner.name}.json"
            else:
                filename = "model.json"

        # Use the storage directory if set, otherwise use current directory

        # Create directory if it doesn't exist
        os.makedirs(self.dir, exist_ok=True)

        # Dump the model as JSON
        filepath = self.dir / filename
        with open(filepath, "w") as f:
            # Exclude any Storage fields to avoid circular references
            model_dict = self.owner.model_dump()
            # Find and remove storage fields to avoid circular references
            f.write(json.dumps(model_dict, indent=2, default=str))

        print(f"Model dumped to {filepath}")
        return filepath

    # This ensures the owner isn't included in serialization
    model_config = ConfigDict(arbitrary_types_allowed=True)


# Make your metaclass inherit from Pydantic's metaclass
class MyMetaclass(pydantic_metaclass):
    def __new__(mcs, name, bases, attrs):
        # Check for annotations in the class attributes
        print(attrs)
        annotations = attrs.get("__annotations__", {})

        # Process annotations to find Storage fields
        for attr_name, attr_type in annotations.items():
            # Check if this is an Annotated type
            print(attr_name, attr_type)
            if get_origin(attr_type) is Annotated:
                type_args = get_args(attr_type)
                base_type = type_args[0]
                metadata = type_args[1:]

                if issubclass(base_type, Storage):
                    if not metadata or isinstance(metadata[0], tuple):
                        raise ValueError(f"no metadata for {metadata}")
                    dir_name = metadata[0]
        #
        if "model_post_init" in attrs:
            raise ValueError(f"{name} has already a model_post_init")

        def new_init(self, c: Any):
            # Now set up all storage fields
            storage = getattr(self, "storage")
            if not storage:
                self.storage = Storage(dir=dir_name, owner=self)

        attrs["model_post_init"] = new_init

        # Call the parent metaclass's __new__ method
        return super().__new__(mcs, name, bases, attrs)


ff = "my-c"


# Now use Annotated in field annotations with "dep" as the directory name
class MyClass(BaseModel, metaclass=MyMetaclass):
    name: str
    age: int = Field(exclude=True)
    storage: Annotated[Storage, ff] = Field(None, exclude=True)


mc = MyClass(name="3da", age=2)
print(mc)

mc.storage.dump()  # This will save "3da.json" to the "dep" directory
