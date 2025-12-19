"""
SerpAPI Google Flights client for searching flight offers.
Alternative to Amadeus API with more comprehensive flight data.
"""
import os
from datetime import datetime
from typing import Dict, List, Optional
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()


class SerpAPIFlightClient:
    """
    Client for interacting with SerpAPI's Google Flights API.

    Features:
    - Direct access to Google Flights data
    - No OAuth required - simple API key
    - Rich flight data including prices, airlines, and layover information
    - Supports one-way and round-trip searches
    """

    def __init__(self):
        """Initialize SerpAPI client with API key from environment or Streamlit secrets."""
        # Try Streamlit secrets first (for Streamlit Cloud), then fall back to .env
        self.api_key = None
        try:
            import streamlit as st
            if "SERPAPI_API_KEY" in st.secrets:
                self.api_key = st.secrets["SERPAPI_API_KEY"]
        except (ImportError, FileNotFoundError, AttributeError, KeyError):
            pass

        # Fall back to environment variable
        if not self.api_key:
            self.api_key = os.getenv("SERPAPI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "SerpAPI API key not found. "
                "Please set SERPAPI_API_KEY in Streamlit secrets or .env file. "
                "Get your API key from https://serpapi.com/manage-api-key"
            )

        print(f"✓ SerpAPI Flight client initialized")

    def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        adults: int = 1,
        max_results: int = 50,
        currency: str = "USD",
        stops: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search for flight offers using SerpAPI Google Flights.

        Args:
            origin: Airport IATA code or city (e.g., "JFK" or "New York")
            destination: Airport IATA code or city (e.g., "LAX" or "Los Angeles")
            departure_date: Departure date in YYYY-MM-DD format
            adults: Number of adult passengers (default: 1)
            max_results: Maximum number of results to return (default: 50)
            currency: Currency for prices (default: "USD")
            stops: Filter by number of stops (None=all, 0=nonstop, 1=1 stop, etc.)

        Returns:
            List of flight offer dictionaries

        Raises:
            Exception if API request fails
        """
        print(f"🔍 Searching flights: {origin} → {destination} on {departure_date}")

        # Prepare search parameters
        params = {
            "engine": "google_flights",
            "departure_id": origin.upper(),
            "arrival_id": destination.upper(),
            "outbound_date": departure_date,
            "type": "2",  # 1 = round trip, 2 = one way, 3 = multi-city
            "currency": currency,
            "hl": "en",
            "api_key": self.api_key,
            "deep_search": "true",  # Enable deep search for more accurate results matching Google Flights browser
        }

        # Add number of adults if > 1
        if adults > 1:
            params["adults"] = adults

        # Add stops filter if specified
        if stops is not None:
            params["stops"] = stops

        try:
            # Make API request
            search = GoogleSearch(params)
            results = search.get_dict()

            # Check for errors
            if "error" in results:
                raise Exception(f"SerpAPI error: {results['error']}")

            # Extract flight offers
            best_flights = results.get("best_flights", [])
            other_flights = results.get("other_flights", [])

            # Combine all flights
            all_flights = best_flights + other_flights

            # Limit to max_results
            all_flights = all_flights[:max_results]

            print(f"✓ Found {len(all_flights)} flight offers ({len(best_flights)} best, {len(other_flights)} other)")

            return all_flights

        except Exception as e:
            raise Exception(f"Failed to search flights via SerpAPI: {str(e)}")

    def parse_flight_offer(self, offer: Dict) -> Dict:
        """
        Parse SerpAPI Google Flights response into simplified format compatible with existing code.

        Args:
            offer: Raw flight offer dictionary from SerpAPI

        Returns:
            Simplified flight data dictionary matching Amadeus format
        """
        try:
            # Extract flight information from SerpAPI format
            flights = offer.get("flights", [])

            if not flights:
                return None

            # First and last flight for overall journey
            first_flight = flights[0]
            last_flight = flights[-1]

            # Extract basic information
            origin = first_flight.get("departure_airport", {}).get("id", "")
            destination = last_flight.get("arrival_airport", {}).get("id", "")

            # Departure and arrival times
            departure_time = first_flight.get("departure_airport", {}).get("time", "")
            arrival_time = last_flight.get("arrival_airport", {}).get("time", "")

            # Duration in minutes
            duration_minutes = offer.get("total_duration", 0)

            # Convert duration to ISO 8601 format (PT1H30M) for compatibility
            hours = duration_minutes // 60
            minutes = duration_minutes % 60
            duration_iso = f"PT{hours}H{minutes}M" if hours > 0 else f"PT{minutes}M"

            # Calculate stops (number of flights - 1)
            stops = len(flights) - 1

            # Extract price
            price = float(offer.get("price", 0))

            # Extract airline information
            airline = first_flight.get("airline", "")
            airline_logo = first_flight.get("airline_logo", "")
            flight_number = first_flight.get("flight_number", "")

            # Layovers information
            layovers = []
            for i in range(len(flights) - 1):
                current_arrival = flights[i].get("arrival_airport", {})
                next_departure = flights[i + 1].get("departure_airport", {})
                layover_duration = flights[i].get("layover", {}).get("duration", 0)

                layovers.append({
                    "airport": current_arrival.get("id", ""),
                    "name": current_arrival.get("name", ""),
                    "duration": layover_duration
                })

            # Additional useful information
            carbon_emissions = offer.get("carbon_emissions", {})

            return {
                "id": offer.get("departure_token", ""),  # Unique identifier
                "origin": origin,
                "destination": destination,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "duration": duration_iso,
                "duration_min": duration_minutes,
                "stops": stops,
                "price": price,
                "airline": airline,
                "airline_logo": airline_logo,
                "flight_number": flight_number,
                "carrier_code": first_flight.get("airline", ""),  # For compatibility
                "layovers": layovers,
                "carbon_emissions": carbon_emissions.get("this_flight", 0),
                "often_delayed": first_flight.get("often_delayed_by_over_30_min", False),
                "extensions": offer.get("extensions", []),
                "raw_data": offer,
            }

        except Exception as e:
            print(f"⚠ Error parsing SerpAPI flight offer: {str(e)}")
            return None

    def get_airline_names(self, airline_codes: List[str]) -> Dict[str, str]:
        """
        Get airline names from codes.

        Note: SerpAPI already provides airline names in the flight data,
        so this method returns a simple mapping for compatibility.

        Args:
            airline_codes: List of airline codes or names

        Returns:
            Dictionary mapping codes to names (identity mapping for SerpAPI)
        """
        # SerpAPI provides full airline names, so just return identity mapping
        return {code: code for code in airline_codes}


if __name__ == "__main__":
    # Test the client
    client = SerpAPIFlightClient()

    # Test search
    flights = client.search_flights(
        origin="JFK",
        destination="LAX",
        departure_date="2025-01-15",
        adults=1,
        max_results=10
    )

    print(f"\n📊 Retrieved {len(flights)} flights")

    # Parse and display first flight
    if flights:
        parsed = client.parse_flight_offer(flights[0])
        if parsed:
            print(f"\n✈️ First flight:")
            print(f"  {parsed['airline']} {parsed['flight_number']}")
            print(f"  {parsed['origin']} → {parsed['destination']}")
            print(f"  ${parsed['price']} | {parsed['duration_min']} min | {parsed['stops']} stops")
