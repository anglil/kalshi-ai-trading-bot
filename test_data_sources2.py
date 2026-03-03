import asyncio, sys
sys.path.insert(0, '.')

async def test_data():
    print("=" * 60)
    print("DATA SOURCE VERIFICATION FOR ALL CONSENSUS STRATEGIES")
    print("=" * 60)
    
    # 4. Econ
    print(f"\n--- ECON ---")
    from src.clients.econ_forecast_client import fetch_econ_forecasts, ECON_INDICATORS
    for ind_name in ECON_INDICATORS:
        try:
            forecast = await fetch_econ_forecasts(ind_name)
            status = "PASS" if len(forecast.sources) >= 3 else ("PARTIAL" if len(forecast.sources) >= 2 else "FAIL")
            print(f'  [{status}] {ind_name}: {len(forecast.sources)} sources, failed={forecast.failed_sources}')
            for s in forecast.sources:
                print(f'         {s.provider}: {s.value:.4f} ({s.model_name})')
        except Exception as e:
            print(f'  [ERROR] {ind_name}: {e}')
    
    # 5. Flu
    print(f"\n--- FLU ---")
    from src.clients.flu_forecast_client import fetch_flu_forecasts
    try:
        flu = await fetch_flu_forecasts()
        status = "PASS" if len(flu.sources) >= 3 else ("PARTIAL" if len(flu.sources) >= 2 else "FAIL")
        print(f'  [{status}] National ILI: {len(flu.sources)} sources, failed={flu.failed_sources}')
        for s in flu.sources:
            print(f'         {s.provider}: ILI={s.value:.2f}% ({s.model_name})')
    except Exception as e:
        print(f'  [ERROR] Flu: {e}')
    
    # 6. Gas
    print(f"\n--- GAS ---")
    from src.clients.gas_forecast_client import fetch_gas_forecasts
    try:
        gas = await fetch_gas_forecasts()
        status = "PASS" if len(gas.sources) >= 3 else ("PARTIAL" if len(gas.sources) >= 2 else "FAIL")
        print(f'  [{status}] Gas Price: {len(gas.sources)} sources, failed={gas.failed_sources}')
        for s in gas.sources:
            print(f'         {s.provider}: ${s.value:.3f} ({s.model_name})')
    except Exception as e:
        print(f'  [ERROR] Gas: {e}')

asyncio.run(test_data())
