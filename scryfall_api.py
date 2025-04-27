import requests
from typing import Dict, List, Optional
import time
from cards import CardType, Color
from card_parser import CardParser

class ScryfallAPI:
    BASE_URL = "https://api.scryfall.com"
    
    def __init__(self):
        self.session = requests.Session()
        # Add a small delay between requests to be nice to Scryfall's servers
        self.session.headers.update({
            'User-Agent': 'MTG_AI_Game_Engine/1.0'
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a request to the Scryfall API with rate limiting."""
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params)
        time.sleep(0.1)  # Rate limiting
        response.raise_for_status()
        return response.json()
    
    def get_card_by_name(self, name: str) -> Dict:
        """Get a card by its exact name."""
        return self._make_request("cards/named", {"exact": name})
    
    def get_card_by_id(self, scryfall_id: str) -> Dict:
        """Get a card by its Scryfall ID."""
        return self._make_request(f"cards/{scryfall_id}")
    
    def search_cards(self, query: str) -> List[Dict]:
        """Search for cards using Scryfall's search syntax."""
        result = self._make_request("cards/search", {"q": query})
        return result.get("data", [])
    
    def get_random_card(self) -> Dict:
        """Get a random card."""
        return self._make_request("cards/random")
    
    def get_set_cards(self, set_code: str) -> List[Dict]:
        """Get all cards from a specific set."""
        result = self._make_request(f"cards/search", {"q": f"set:{set_code}"})
        return result.get("data", [])
    
    def convert_to_card_data(self, scryfall_card: Dict) -> Dict:
        """Convert Scryfall card data to our internal format."""
        # Parse oracle text into abilities
        abilities = CardParser.parse_oracle_text(scryfall_card.get("oracle_text", ""))
        
        # Basic card data
        card_data = {
            "name": scryfall_card["name"],
            "mana_cost": scryfall_card.get("mana_cost", ""),
            "card_type": self._determine_card_type(scryfall_card.get("type_line", "")),
            "colors": [color.upper() for color in scryfall_card.get("colors", [])],
            "abilities": abilities
        }
        
        # Add type-specific data
        if "creature" in scryfall_card.get("type_line", "").lower():
            card_data.update({
                "power": scryfall_card.get("power", "0"),
                "toughness": scryfall_card.get("toughness", "0")
            })
            
        return card_data
    
    def _determine_card_type(self, type_line: str) -> str:
        """Convert Scryfall type line to our CardType enum."""
        type_line = type_line.upper()
        if "CREATURE" in type_line:
            return "CREATURE"
        elif "LAND" in type_line:
            return "LAND"
        elif "INSTANT" in type_line:
            return "INSTANT"
        elif "SORCERY" in type_line:
            return "SORCERY"
        elif "ARTIFACT" in type_line:
            return "ARTIFACT"
        elif "ENCHANTMENT" in type_line:
            return "ENCHANTMENT"
        elif "PLANESWALKER" in type_line:
            return "PLANESWALKER"
        else:
            return "CREATURE"  # Default fallback 