"""
Amadeus API client for searching flight offers.
Handles OAuth2 authentication and flight search requests.
"""
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()


class AmadeusClient:
    """
    Client for interacting with Amadeus Flight Offers Search API v2.

    Features:
    - OAuth2 token management with automatic refresh
    - Flight search with comprehensive parameters
    - Error handling and retry logic
    - Supports both test and production environments
    """

    def __init__(self):
        """Initialize Amadeus client with credentials from environment or Streamlit secrets."""
        # Try Streamlit secrets first (for Streamlit Cloud), then fall back to .env
        try:
            import streamlit as st
            self.api_key = st.secrets.get("AMADEUS_API_KEY", os.getenv("AMADEUS_API_KEY"))
            self.api_secret = st.secrets.get("AMADEUS_API_SECRET", os.getenv("AMADEUS_API_SECRET"))
            base_url = st.secrets.get("AMADEUS_BASE_URL", os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com"))
        except (ImportError, FileNotFoundError, AttributeError):
            # Not running in Streamlit or secrets not configured, use environment variables
            self.api_key = os.getenv("AMADEUS_API_KEY")
            self.api_secret = os.getenv("AMADEUS_API_SECRET")
            base_url = os.getenv("AMADEUS_BASE_URL", "https://test.api.amadeus.com")

        # Remove trailing slash if present
        base_url = base_url.rstrip('/')

        # Set API endpoints based on base URL
        self.AUTH_URL = f"{base_url}/v1/security/oauth2/token"
        self.FLIGHT_OFFERS_URL = f"{base_url}/v2/shopping/flight-offers"
        self.AIRLINE_LOOKUP_URL = f"{base_url}/v1/reference-data/airlines"

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Amadeus API credentials not found. "
                "Please set AMADEUS_API_KEY and AMADEUS_API_SECRET in .env file"
            )

        self.access_token = None
        self.token_expires_at = None

        # Log which environment we're using
        environment = "PRODUCTION" if "api.amadeus.com" in base_url else "TEST"
        print(f"✓ Amadeus client initialized ({environment} environment)")
        print(f"  Base URL: {base_url}")

    def _get_access_token(self) -> str:
        """
        Obtain OAuth2 access token from Amadeus API.
        Tokens are cached and reused until expiration.

        Returns:
            Access token string

        Raises:
            Exception if authentication fails
        """
        # Return cached token if still valid
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token

        # Request new token
        try:
            response = requests.post(
                self.AUTH_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.api_key,
                    "client_secret": self.api_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            self.access_token = data["access_token"]

            # Set expiration time (subtract 60 seconds for safety margin)
            expires_in = data.get("expires_in", 1799)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

            print(f"✓ Amadeus access token obtained (expires in {expires_in}s)")
            return self.access_token

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to obtain Amadeus access token: {str(e)}")

    def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        adults: int = 1,
        max_results: int = 10,
        currency_code: str = "USD",
        non_stop: bool = False,
    ) -> List[Dict]:
        """
        Search for flight offers using Amadeus API.

        Args:
            origin: IATA code of origin airport (e.g., "JFK")
            destination: IATA code of destination airport (e.g., "LAX")
            departure_date: Departure date in YYYY-MM-DD format
            adults: Number of adult passengers (default: 1)
            max_results: Maximum number of results to return (default: 10, max: 250)
            currency_code: Currency for prices (default: "USD")
            non_stop: Only return non-stop flights if True (default: False)

        Returns:
            List of flight offer dictionaries from Amadeus API

        Raises:
            Exception if API request fails
        """
        # Get access token
        token = self._get_access_token()

        # Prepare request parameters
        params = {
            "originLocationCode": origin.upper(),
            "destinationLocationCode": destination.upper(),
            "departureDate": departure_date,
            "adults": adults,
            "max": min(max_results, 250),  # API limit is 250
            "currencyCode": currency_code,
        }

        if non_stop:
            params["nonStop"] = "true"

        # Make API request
        try:
            response = requests.get(
                self.FLIGHT_OFFERS_URL,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            data = response.json()
            flight_offers = data.get("data", [])

            print(
                f"✓ Found {len(flight_offers)} flight offers for {origin} -> {destination} on {departure_date}"
            )

            return flight_offers

        except requests.exceptions.HTTPError as e:
            error_msg = f"Amadeus API request failed: {e.response.status_code}"
            try:
                error_detail = e.response.json()
                error_msg += f" - {error_detail}"
            except Exception:
                error_msg += f" - {e.response.text}"
            raise Exception(error_msg)

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to search flights: {str(e)}")

    def search_airports(self, keyword: str, max_results: int = 10) -> List[Dict]:
        """
        Search for airports by keyword (city name or IATA code) using Amadeus locations API.

        Args:
            keyword: City name or airport code to search (e.g., "New York", "JFK")
            max_results: Max results to return (default 10)

        Returns:
            List of dicts with keys: iata_code, name, city, country, label
        """
        token = self._get_access_token()
        url = self.AUTH_URL.replace("/v1/security/oauth2/token", "/v1/reference-data/locations")

        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "keyword": keyword,
                    "subType": "AIRPORT",
                    "page[limit]": max_results,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            results = []
            for loc in data.get("data", []):
                iata = loc.get("iataCode", "")
                name = loc.get("name", "")
                city = loc.get("address", {}).get("cityName", "")
                country = loc.get("address", {}).get("countryName", "")
                label = f"{iata} - {name}, {city}, {country}"
                results.append({"iata_code": iata, "name": name, "city": city, "country": country, "label": label})
            return results
        except requests.exceptions.RequestException as e:
            print(f"⚠ Airport search failed: {str(e)}")
            return []

    def get_airport_coordinates(self, iata_code: str) -> Optional[Tuple[float, float]]:
        """
        Look up airport coordinates by IATA code.

        Args:
            iata_code: Airport code (e.g., "TPA")

        Returns:
            (latitude, longitude) if found, else None
        """
        token = self._get_access_token()
        code = (iata_code or "").strip().upper()
        if not code:
            return None

        url = self.AUTH_URL.replace("/v1/security/oauth2/token", "/v1/reference-data/locations")

        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "keyword": code,
                    "subType": "AIRPORT",
                    "page[limit]": 10,
                },
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            for loc in data.get("data", []):
                if (loc.get("iataCode") or "").upper() != code:
                    continue
                geo = loc.get("geoCode", {})
                lat = geo.get("latitude")
                lon = geo.get("longitude")
                if lat is None or lon is None:
                    continue
                return float(lat), float(lon)

            return None

        except requests.exceptions.RequestException as e:
            print(f"⚠ Airport coordinate lookup failed for {code}: {str(e)}")
            return None

    def search_nearby_airports(
        self,
        latitude: float,
        longitude: float,
        radius_miles: int = 100,
        max_results: int = 30,
    ) -> List[Dict]:
        """
        Search airports near a coordinate using Amadeus geospatial endpoint.

        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            radius_miles: Search radius in miles (default 100)
            max_results: Max airports to return

        Returns:
            List of dicts with keys: iata_code, name, city, country, distance_miles
        """
        token = self._get_access_token()
        url = self.AUTH_URL.replace("/v1/security/oauth2/token", "/v1/reference-data/locations/airports")

        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "radius": radius_miles,
                    "radiusUnit": "MILE",
                    "sort": "distance",
                    "page[limit]": max_results,
                },
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            results = []
            for loc in data.get("data", []):
                iata = loc.get("iataCode", "")
                if not iata:
                    continue
                name = loc.get("name", "")
                city = loc.get("address", {}).get("cityName", "")
                country = loc.get("address", {}).get("countryName", "")
                distance_val = (loc.get("distance") or {}).get("value")
                distance_miles = float(distance_val) if distance_val is not None else None
                results.append(
                    {
                        "iata_code": iata,
                        "name": name,
                        "city": city,
                        "country": country,
                        "distance_miles": distance_miles,
                    }
                )
            return results

        except requests.exceptions.RequestException as e:
            print(f"⚠ Nearby airport search failed ({latitude}, {longitude}): {str(e)}")
            return []

    def expand_airports_within_radius(
        self,
        airport_codes: List[str],
        radius_miles: int = 100,
        max_results_per_airport: int = 30,
    ) -> List[str]:
        """
        Expand IATA codes to include nearby airports in a radius.

        Args:
            airport_codes: Seed airport codes from parser
            radius_miles: Radius around each seed airport
            max_results_per_airport: Max nearby airports per seed

        Returns:
            Deduplicated list of IATA codes including nearby airports
        """
        expanded_codes: List[str] = []
        seen = set()

        for raw_code in airport_codes or []:
            code = (raw_code or "").strip().upper()
            if not code:
                continue

            if code not in seen:
                expanded_codes.append(code)
                seen.add(code)

            coords = self.get_airport_coordinates(code)
            if not coords:
                continue

            nearby = self.search_nearby_airports(
                latitude=coords[0],
                longitude=coords[1],
                radius_miles=radius_miles,
                max_results=max_results_per_airport,
            )

            for airport in nearby:
                nearby_code = (airport.get("iata_code") or "").strip().upper()
                if nearby_code and nearby_code not in seen:
                    expanded_codes.append(nearby_code)
                    seen.add(nearby_code)

        return expanded_codes

    def get_airline_names(self, airline_codes: List[str]) -> Dict[str, str]:
        """
        Look up airline names from IATA codes using Amadeus API.

        Args:
            airline_codes: List of IATA airline codes (e.g., ["AA", "DL", "UA"])

        Returns:
            Dictionary mapping airline codes to names (e.g., {"AA": "American Airlines"})
        """
        if not airline_codes:
            return {}

        # Get access token
        token = self._get_access_token()

        # Build comma-separated list of codes
        codes_param = ",".join(set(airline_codes))  # Remove duplicates

        try:
            response = requests.get(
                self.AIRLINE_LOOKUP_URL,
                headers={"Authorization": f"Bearer {token}"},
                params={"airlineCodes": codes_param},
                timeout=10,
            )
            response.raise_for_status()

            data = response.json()
            airlines = data.get("data", [])

            # Build mapping of code to name
            airline_map = {}
            for airline in airlines:
                code = airline.get("iataCode")
                name = airline.get("businessName") or airline.get("commonName")
                if code and name:
                    airline_map[code] = name

            print(f"✓ Looked up {len(airline_map)} airline names")
            return airline_map

        except requests.exceptions.RequestException as e:
            print(f"⚠ Failed to lookup airline names: {str(e)}")
            # Return empty dict on failure, fallback to codes
            return {}

    def parse_flight_offer(self, offer: Dict) -> Dict:
        """
        Parse Amadeus flight offer response into simplified format.

        Args:
            offer: Raw flight offer dictionary from Amadeus API

        Returns:
            Simplified flight data dictionary
        """
        try:
            # Extract itinerary (first itinerary for one-way flights)
            itinerary = offer.get("itineraries", [{}])[0]
            segments = itinerary.get("segments", [])

            if not segments:
                return None

            # First and last segment
            first_segment = segments[0]
            last_segment = segments[-1]

            # Extract basic information
            origin = first_segment.get("departure", {}).get("iataCode")
            destination = last_segment.get("arrival", {}).get("iataCode")
            departure_time = first_segment.get("departure", {}).get("at")
            arrival_time = last_segment.get("arrival", {}).get("at")
            duration = itinerary.get("duration")

            # Calculate stops
            stops = len(segments) - 1

            # Extract layover airports (intermediate stops)
            layover_airports = [
                seg.get("arrival", {}).get("iataCode")
                for seg in segments[:-1]
                if seg.get("arrival", {}).get("iataCode")
            ]

            # Extract price
            price = float(offer.get("price", {}).get("total", 0))

            # Extract airline name (from first segment)
            carrier_code = first_segment.get("carrierCode", "")

            # Extract flight number (e.g., "AA123")
            flight_num = first_segment.get("number", "")
            flight_number = f"{carrier_code}{flight_num}" if flight_num else carrier_code

            # Convert duration from format like "PT2H30M" to minutes
            duration_minutes = self._parse_duration_to_minutes(duration)

            # Extract cabin class and checked bags from travelerPricings
            cabin = None
            checked_bags = 0
            traveler_pricings = offer.get("travelerPricings", [])
            if traveler_pricings:
                fare_details = traveler_pricings[0].get("fareDetailsBySegment", [])
                if fare_details:
                    cabin = fare_details[0].get("cabin")
                    included_bags = fare_details[0].get("includedCheckedBags", {})
                    checked_bags = included_bags.get("quantity", 0) if included_bags else 0

            return {
                "id": offer.get("id"),  # Amadeus offer ID
                "origin": origin,
                "destination": destination,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
                "duration": duration,  # Keep original format
                "duration_min": duration_minutes,  # Add SerpAPI-compatible field
                "stops": stops,
                "layover_airports": layover_airports,
                "price": price,
                "carrier_code": carrier_code,
                "airline": carrier_code,  # Alias for compatibility with SerpAPI format
                "airline_name": carrier_code,  # Add airline_name field
                "flight_number": flight_number,  # Flight number (e.g., "AA123")
                "cabin": cabin,          # e.g. ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST
                "checked_bags": checked_bags,  # Number of included checked bags
                "raw_data": offer,
            }

        except Exception as e:
            print(f"Error parsing flight offer: {str(e)}")
            return None

    def _parse_duration_to_minutes(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration (e.g., 'PT2H30M') to total minutes.

        Args:
            duration_str: Duration in ISO 8601 format

        Returns:
            Total duration in minutes
        """
        try:
            import re
            hours = 0
            minutes = 0

            # Extract hours if present (e.g., "2H")
            hour_match = re.search(r'(\d+)H', duration_str)
            if hour_match:
                hours = int(hour_match.group(1))

            # Extract minutes if present (e.g., "30M")
            minute_match = re.search(r'(\d+)M', duration_str)
            if minute_match:
                minutes = int(minute_match.group(1))

            return hours * 60 + minutes
        except Exception:
            return 0
