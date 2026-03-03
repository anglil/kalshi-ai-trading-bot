import asyncio, sys
sys.path.insert(0, '.')

async def test_data():
    import aiohttp
    
    print("=" * 60)
    print("DATA SOURCE VERIFICATION FOR ALL CONSENSUS STRATEGIES")
    print("=" * 60)
    
    # 1. Weather: 3 sources (NWS, GFS, ECMWF)
    from src.clients.weather_forecast_client import fetch_all_forecasts
    from src.clients.nws_client import WEATHER_STATIONS
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\n--- WEATHER (target date: {today}) ---")
    for city_key, station in WEATHER_STATIONS.items():
        forecast = await fetch_all_forecasts(station, today)
        status = "PASS" if len(forecast.sources) >= 3 else "FAIL"
        print(f'  [{status}] {station.city}: {len(forecast.sources)}/3 sources, failed={forecast.failed_sources}')
        for s in forecast.sources:
            print(f'         {s.provider}: {s.temperature_high_f:.1f}F ({s.model_name})')
    
    # 2. NBA
    print(f"\n--- NBA ---")
    from src.clients.nba_odds_client import fetch_nba_forecasts
    nba = await fetch_nba_forecasts(odds_api_key='')
    print(f'  Total games: {len(nba)}')
    for g in nba:
        status = "PASS" if len(g.sources) >= 2 else "FAIL"
        print(f'  [{status}] {g.home_team} vs {g.away_team}: {len(g.sources)} sources')
        for s in g.sources:
            print(f'         {s.provider}: H={s.home_win:.0%} A={s.away_win:.0%}')
    
    # 3. Soccer
    print(f"\n--- SOCCER ---")
    from src.clients.soccer_odds_client import fetch_soccer_forecasts
    soccer = await fetch_soccer_forecasts(odds_api_key='', api_football_key='')
    print(f'  Total matches: {len(soccer)}')
    for g in soccer:
        status = "PASS" if len(g.sources) >= 2 else "FAIL"
        print(f'  [{status}] {g.home_team} vs {g.away_team} ({g.league}): {len(g.sources)} sources')
        for s in g.sources:
            print(f'         {s.provider}: H={s.home_win:.0%} D={s.draw:.0%} A={s.away_win:.0%}')
    
    # 4. Econ
    print(f"\n--- ECON ---")
    from src.clients.econ_forecast_client import fetch_econ_forecasts
    econ = await fetch_econ_forecasts()
    print(f'  Indicators: {len(econ)}')
    for ind_name, forecast in econ.items():
        status = "PASS" if len(forecast.sources) >= 3 else ("PARTIAL" if len(forecast.sources) >= 2 else "FAIL")
        print(f'  [{status}] {ind_name}: {len(forecast.sources)} sources, failed={forecast.failed_sources}')
        for s in forecast.sources:
            print(f'         {s.provider}: {s.value:.4f} ({s.model_name})')
    
    # 5. Flu
    print(f"\n--- FLU ---")
    from src.clients.flu_forecast_client import fetch_flu_forecasts
    flu = await fetch_flu_forecasts()
    status = "PASS" if len(flu.sources) >= 3 else ("PARTIAL" if len(flu.sources) >= 2 else "FAIL")
    print(f'  [{status}] National ILI: {len(flu.sources)} sources, failed={flu.failed_sources}')
    for s in flu.sources:
        print(f'         {s.provider}: ILI={s.value:.2f}% ({s.model_name})')
    
    # 6. Gas
    print(f"\n--- GAS ---")
    from src.clients.gas_forecast_client import fetch_gas_forecasts
    gas = await fetch_gas_forecasts()
    status = "PASS" if len(gas.sources) >= 3 else ("PARTIAL" if len(gas.sources) >= 2 else "FAIL")
    print(f'  [{status}] Gas Price: {len(gas.sources)} sources, failed={gas.failed_sources}')
    for s in gas.sources:
        print(f'         {s.provider}: ${s.value:.3f} ({s.model_name})')

asyncio.run(test_data())
