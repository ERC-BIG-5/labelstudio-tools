from calendar import monthrange

if __name__ == '__main__':
    total_days = 0
    years = [2022, 2023]
    platforms = ["twitter", "youtube", "tiktok"]
    languages = ["en", "es"]
    print(f"{years=}")
    print(f"{platforms=}")
    print(f"{languages=}")
    for y in years:
        for m in range(1, 13):
            days = monthrange(y, m)[1]
            # print(y,m,days)
            total_days += days
    print("***")
    print(f"{years=}, {total_days=}")
    num_platforms = len(platforms)
    num_languages = len(languages)
    plat_lang = len(platforms) * len(languages)
    print(f"plat_lang= {num_platforms * num_languages=}")
    d_post_per_day_result = plat_lang * total_days
    # p_d = [1, 2, 3]
    # for p in p_d:
    #     print(f"{p} post per day: {d_post_per_day_result * p}")
    total_hours = total_days * 24
    post_every_hours = 10
    print(f"{total_hours=}; {post_every_hours=}")
    print(f"{total_hours / post_every_hours * plat_lang=}")
