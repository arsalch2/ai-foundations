"""Tools the model may call. The inventory is fake; in project 2 this becomes an MCP server over a mock ERP."""

from pydantic import BaseModel, Field

# Fake warehouse data
INVENTORY: dict[str, dict] = {
    "6205-2RS": {"name": "Deep groove ball bearing 6205-2RS", "in_stock": 1240, "lead_time_days": 3},
    "6305-ZZ": {"name": "Deep groove ball bearing 6305-ZZ", "in_stock": 80, "lead_time_days": 10},
    "22210-E": {"name": "Spherical roller bearing 22210-E", "in_stock": 0, "lead_time_days": 21},
}


class CheckStock(BaseModel):
    """Look up stock level and lead time for one article number."""

    article_number: str = Field(description="Article number, e.g. 6205-2RS")


def check_stock(args: CheckStock) -> dict:
    item = INVENTORY.get(args.article_number.strip().upper())
    if item is None:
        return {"found": False, "article_number": args.article_number}
    return {"found": True, "article_number": args.article_number, **item}


# name -> (argument model, function)
TOOLS = {
    "CheckStock": (CheckStock, check_stock),
}
