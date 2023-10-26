class MyArgs:
    def __init__(self, model_path, model_id,  num_gpus_per_model, num_gpus_total, if_ref_sup, if_ref_drop, if_fast_eval=1, question_file="/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/mix-instruct/val_data_prepared_sampled_0627.jsonl", answer_file=None, reference_file="/home/zhulianghui/ProjectC_ChatGPT/alpaca/reference/mix-instruct/val_data_references.jsonl", temperature=0.2):
        self.model_path = model_path
        self.model_id = model_id
        self.question_file = question_file
        self.num_gpus_per_model = num_gpus_per_model
        self.num_gpus_total = num_gpus_total
        self.if_ref_sup = if_ref_sup
        self.if_ref_drop = if_ref_drop
        self.if_fast_eval = if_fast_eval
        self.answer_file = answer_file
        self.reference_file = reference_file
        self.temperature = temperature

if __name__ == '__main__':

    # 手动创建一个 MyArgs 对象，模拟通过命令行传递的参数
    my_args = MyArgs(
        model_path="/path/to/model",
        model_id="12345",
        question_file="questions.txt",
        answer_file="answers.txt",
        num_choices=4
    )

    # 现在，你可以像正常的 args 一样使用 my_args 对象的属性
    print(my_args.model_path)
    print(my_args.model_id)
    print(my_args.question_file)
    print(my_args.answer_file)
    print(my_args.num_choices)

    if False:
        print("start to eval!")
        from judgelm.modules.myargs import MyArgs
        from judgelm.llm_judge.eval_model_review_answer import eval_judgelm_performance_all

        myargs = MyArgs(model_path=training_args.output_dir,
                        model_id=training_args.run_name,
                        num_gpus_per_model=1 if (
                                    "7b" in model_args.model_name_or_path or "13b" in model_args.model_name_or_path) else 2,
                        num_gpus_total=8,
                        if_ref_sup=data_args.ref_drop_ratio > 1.5,
                        if_ref_drop=data_args.ref_drop_ratio > -0.5 and data_args.ref_drop_ratio <= 1.0,
                        )

        # turn `--if_w_reference` in myargs into a boolean
        myargs.if_ref_sup = bool(myargs.if_ref_sup)
        myargs.if_ref_drop = bool(myargs.if_ref_drop)

        myargs.if_fast_eval = bool(myargs.if_fast_eval)

        # turn `--reference-file` in myargs into boolean or not
        if myargs.reference_file == 'None':
            myargs.reference_file = None

        print(f"myargs: {myargs}")

        if myargs.num_gpus_total // myargs.num_gpus_per_model > 1:
            import ray

            print("ray has been improted")
            ray.init(num_cpus=int(myargs.num_gpus_total / myargs.num_gpus_per_model))
            # ray.init()
            print("ray has been initialized")

        results = eval_judgelm_performance_all(myargs)
        print("len of results: ", len(results))
        for result in results:
            print(result)
