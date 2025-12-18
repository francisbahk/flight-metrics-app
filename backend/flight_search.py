"""
Unified flight search interface supporting multiple providers.
Allows easy switching between Amadeus and SerpAPI (Google Flights).
"""
import os
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


class FlightSearchClient:
    """
    Unified client for flight search supporting multiple providers.

    Providers:
    - 'serpapi': SerpAPI Google Flights (recommended, more data)
    - 'amadeus': Amadeus API (legacy, requires OAuth)

    Set FLIGHT_API_PROVIDER environment variable to choose provider.
    Defaults to 'serpapi' if both API keys are available.
    """

    def __init__(self, provider: Optional[str] = None):
        """
        Initialize flight search client with specified provider.

        Args:
            provider: 'serpapi' or 'amadeus'. If None, auto-detect from environment.
        """
        # Determine provider
        if provider is None:
            provider = os.getenv("FLIGHT_API_PROVIDER", "serpapi").lower()

        self.provider = provider

        # Initialize the appropriate client
        if provider == "serpapi":
            from backend.serpapi_client import SerpAPIFlightClient
            self.client = SerpAPIFlightClient()
            print(f"✓ Using SerpAPI (Google Flights) for flight search")

        elif provider == "amadeus":
            from backend.amadeus_client import AmadeusClient
            self.client = AmadeusClient()
            print(f"✓ Using Amadeus API for flight search")

        else:
            raise ValueError(
                f"Unknown flight API provider: {provider}. "
                f"Valid options: 'serpapi', 'amadeus'"
            )

    def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        adults: int = 1,
        max_results: int = 50,
        currency: str = "USD",
        non_stop: bool = False,
    ) -> List[Dict]:
        """
        Search for flight offers using the configured provider.

        Args:
            origin: Airport IATA code (e.g., "JFK")
            destination: Airport IATA code (e.g., "LAX")
            departure_date: Departure date in YYYY-MM-DD format
            adults: Number of adult passengers (default: 1)
            max_results: Maximum number of results to return (default: 50)
            currency: Currency for prices (default: "USD")
            non_stop: Only return non-stop flights if True (default: False)

        Returns:
            List of flight offer dictionaries from the provider
        """
        if self.provider == "serpapi":
            # SerpAPI uses 'stops' parameter differently
            stops = 0 if non_stop else None
            return self.client.search_flights(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                adults=adults,
                max_results=max_results,
                currency=currency,
                stops=stops,
            )

        elif self.provider == "amadeus":
            # Amadeus uses 'currency_code' and max is capped at 250
            return self.client.search_flights(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                adults=adults,
                max_results=min(max_results, 250),
                currency_code=currency,
                non_stop=non_stop,
            )

    def parse_flight_offer(self, offer: Dict) -> Optional[Dict]:
        """
        Parse flight offer into simplified format.

        Args:
            offer: Raw flight offer dictionary from provider

        Returns:
            Simplified flight data dictionary
        """
        return self.client.parse_flight_offer(offer)

    def get_airline_names(self, airline_codes: List[str]) -> Dict[str, str]:
        """
        Look up airline names from codes.

        Args:
            airline_codes: List of airline codes

        Returns:
            Dictionary mapping codes to names
        """
        return self.client.get_airline_names(airline_codes)


# Convenience function for backwards compatibility
def get_flight_client() -> FlightSearchClient:
    """
    Get a flight search client with auto-detected provider.

    Returns:
        FlightSearchClient instance
    """
    return FlightSearchClient()


if __name__ == "__main__":
    # Test the unified client
    print("Testing unified flight search client\n")

    # Test with SerpAPI
    try:
        client = FlightSearchClient(provider="serpapi")
        flights = client.search_flights(
            origin="JFK",
            destination="LAX",
            departure_date="2025-01-15",
            max_results=5
        )
        print(f"\n✓ SerpAPI: Found {len(flights)} flights")

        if flights:
            parsed = client.parse_flight_offer(flights[0])
            if parsed:
                print(f"  First flight: {parsed['airline']} - ${parsed['price']}")
    except Exception as e:
        print(f"\n✗ SerpAPI test failed: {e}")

    # Test with Amadeus (commented out to avoid errors if not configured)
    # try:
    #     client = FlightSearchClient(provider="amadeus")
    #     flights = client.search_flights(
    #         origin="JFK",
    #         destination="LAX",
    #         departure_date="2025-01-15",
    #         max_results=5
    #     )
    #     print(f"\n✓ Amadeus: Found {len(flights)} flights")
    # except Exception as e:
    #     print(f"\n✗ Amadeus test failed: {e}")
