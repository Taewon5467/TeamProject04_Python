import argparse
from config import DEFAULT_DATE, DEFAULT_WEEKDAY, DEFAULT_TEMP, DEFAULT_RAIN, DEFAULT_SNOW, DEFAULT_VIS
from geocoding      import load_with_geocoding
from clustering     import run_clustering, build_spots, evaluate_clustering
from prediction     import load_pipeline, run_simulation, assign_robots, save_simulation_result
from visualization import generate_html, open_map_in_browser


def parse_args():
    parser = argparse.ArgumentParser(description="배달 로봇 최적 거점 분석")
    parser.add_argument("--date",    default=DEFAULT_DATE)
    parser.add_argument("--weekday", default=DEFAULT_WEEKDAY,
                        choices=["월요일","화요일","수요일","목요일","금요일","토요일","일요일"])
    parser.add_argument("--temp",    type=float, default=DEFAULT_TEMP)
    parser.add_argument("--rain",    type=float, default=DEFAULT_RAIN)
    parser.add_argument("--snow",    type=float, default=DEFAULT_SNOW)
    parser.add_argument("--vis",     type=float, default=DEFAULT_VIS)
    parser.add_argument("--holiday", action="store_true", help="공휴일 여부")
    parser.add_argument("--no-save", action="store_true", help="결과 CSV 저장 생략")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n[1/5] 좌표 변환")
    df_clean = load_with_geocoding()

    print(f"\n[2/5] 군집화")
    df_clean, Z, max_dist_km = run_clustering(df_clean)
    spots = build_spots(df_clean)
    print(f"  → 거점 {len(spots)}개 생성")

    print(f"\n[3/5] ML 예측")
    model, features = load_pipeline()

    if model:
        hourly_demand = run_simulation(model, features,
                                       args.date, args.weekday,
                                       args.temp, args.rain, args.snow, args.vis,
                                       int(args.holiday))
        spots = assign_robots(spots, hourly_demand)
        print(f"  → 피크: {int(hourly_demand.idxmax())}시 ({hourly_demand.max():.1f}건)")
        print(f"  → 평균: {hourly_demand.mean():.1f}건")

        if not args.no_save:
            print(f"\n[4/5] 결과 저장")
            save_simulation_result(hourly_demand, args.date, args.weekday)
    else:
        for spot in spots:
            spot['robots_peak'] = spot['robots_avg'] = '?'

    print(f"\n[5/5] HTML 생성")
    generate_html(df_clean, spots, max_dist_km)

    open_map_in_browser(port=8000)
    
    print(f"  기온: {args.temp} / 강수량: {args.rain}")

    # 서버가 살아있어야 하므로 메인 스레드 유지
    input("\n🌐 서버 실행 중... 종료하려면 Enter를 누르세요.")

    metrics = evaluate_clustering(df_clean, Z, max_dist_km)
    print(f"\n📊 CPCC: {metrics['cpcc']} | 위반 군집: {metrics['violation_count']}개")
    print("✅ 완료\n")


if __name__ == "__main__":
    main()
    
#python -m http.server 8000
#http://localhost:8000/naver_store_map.html