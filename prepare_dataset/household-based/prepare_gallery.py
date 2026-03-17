import json
from collections import defaultdict
from multiprocessing.resource_sharer import stop
# from ..analyze_data.check_data import parse_id, save_json
from tqdm import tqdm
import random


def parse_id(filename):
    return '_'.join(filename.split('/')[-1].split('_')[:-4])

def parse_household_id(identity_id):
    return '_'.join(identity_id.split('_')[:-1])

def parse_mac_addr(item):
    return item.split('_')[-4]

def parse_event_id(item):
    return item.split('_')[-3]


def save_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")


def get_household_info(gallery):
    res = defaultdict(dict)
    for item in gallery:
        identity_id = parse_id(item)
        household_id = parse_household_id(identity_id)
        mac_addr = parse_mac_addr(item)
        res[household_id] = res.get(household_id, dict())
        res[household_id][identity_id] = res[household_id].get(identity_id, dict())
        res[household_id][identity_id][mac_addr] = res[household_id][identity_id].get(mac_addr, [])
        res[household_id][identity_id][mac_addr].append(item)

    return res


def prepare_household_gallery(strategy, household_type, clothes, camera, household_dict, query):

    print(f'\nProcessing case: {strategy}...')
    eval_cases = {} # key is identity_id, value is a list of test cases for this identity_id
    total_eval_ct = 0
    id_ct = defaultdict(int)
    event_ct = defaultdict(int)
    for q in tqdm(query.keys()):
        identity_id = parse_id(q)
        household_id = parse_household_id(identity_id)
        mac_addr = parse_mac_addr(q)
        event_id = parse_event_id(q)

        if household_type == 'single':

            if camera == 'same': # same camera mac address
                if mac_addr not in household_dict[household_id][identity_id]:
                    continue
                positive_pool = household_dict[household_id][identity_id][mac_addr]

            else: # different camera mac address
                positive_pool = []
                for mac in household_dict[household_id][identity_id]:
                    if mac != mac_addr:
                        positive_pool += household_dict[household_id][identity_id][mac]

            # since this is single household, negative pool is randomly sampled from
            # identities from all other households
            negative_pool = []
            for household in household_dict:
                if household != household_id:
                    for identity in household_dict[household]:
                        for mac in household_dict[household][identity]:
                            negative_pool += household_dict[household][identity][mac]

        elif household_type == 'multiple':

            if camera == 'same': # same camera mac address
                if mac_addr not in household_dict[household_id][identity_id]:
                    continue
                positive_pool = household_dict[household_id][identity_id][mac_addr]

                # negatives are from other members in the same household with the same camera mac address
                negative_pool = []
                for identity in household_dict[household_id]:
                    if identity != identity_id:
                        if mac_addr in household_dict[household_id][identity]:
                            negative_pool += household_dict[household_id][identity][mac_addr]

            else: # cross camera, different camera mac address
                positive_pool = []
                for mac in household_dict[household_id][identity_id]:
                    if mac != mac_addr:
                        positive_pool += household_dict[household_id][identity_id][mac]

                # negatives are from other members in the same household with different camera mac address
                negative_pool = []
                for identity in household_dict[household_id]:
                    if identity != identity_id:
                        for mac in household_dict[household_id][identity]:
                            if mac != mac_addr:
                                negative_pool += household_dict[household_id][identity][mac]
        else:
            raise ValueError

        # randomly sample 1 positive example from positive pool
        # and 4 negatives from the negative pool
        g = []
        if len(positive_pool) > 1 and len(negative_pool) > 4:
            id_ct[identity_id] += 1
            event_ct[event_id] += 1
            positive_example = random.choice(positive_pool)
            negative_examples = random.sample(negative_pool, 4)
            g.append((positive_example, 1))
            g += [(neg, 0) for neg in negative_examples]

            # shuffle the gallery examples
            random.shuffle(g)
            # get the label, which is the index of the positive example in the gallery
            label  = g.index((positive_example, 1)) + 1
            if identity_id not in eval_cases:
                eval_cases[identity_id] = []
            eval_cases[identity_id].append({
                'query': q,
                'gallery': [x[0] for x in g],
                'label': label
            })
            total_eval_ct += 1

    print(f"{household_type} household: {total_eval_ct} test cases for clothes={clothes} and camera={camera}.")
    print(f"Identity ct: {len(id_ct)}")
    print(f"Event ct: {len(event_ct)}")

    # sort id_ct and event_ct by value
    id_ct = dict(sorted(id_ct.items(), key=lambda item: item[1], reverse=True))
    event_ct = dict(sorted(event_ct.items(), key=lambda item: item[1], reverse=True))


    ## we do some subsampling here to speed up the evaluation.
    ## We sample max(50, len(id_ct)) identities, and for each identity
    ## we unifromly sample max(3, ct) examples from it original sorted queries.

    res = dict()
    # randomly sample max(50, len(id_ct)) identities
    sampled_id_ct = dict(random.sample(list(id_ct.items()), min(50, len(id_ct))))
    # obtain the event ct for the sampled identities
    sampled_event_ct = defaultdict(int)
    for identity_id in sampled_id_ct:
        for case in eval_cases[identity_id]:
            event_id = parse_event_id(case['query'])
            sampled_event_ct[event_id] += 1
    print(f"After sampling, Identity ct: {len(sampled_id_ct)}")
    print(f"After sampling, Event ct: {len(sampled_event_ct)}")

    # sample eval cases
    sampled_eval_cases = []
    for identity_id in sampled_id_ct:
        cases = eval_cases[identity_id]
        # sort the cases by the query filename
        cases = sorted(cases, key=lambda x: x['query'])

        # let's uniformly sample max(3, ct) examples from the original sorted queries for this identity_id
        # by taking the start, 2 middle frames if there are enough examples, otherwise just take all examples
        if len(cases) > 3:
            sampled_cases = [cases[0], cases[len(cases) // 3], cases[2 * len(cases) // 3]]
        else:
            sampled_cases = cases
        sampled_eval_cases += sampled_cases
    print(f"After sampling, total eval cases: {len(sampled_eval_cases)}")


    res['id_count'] = sampled_id_ct
    # res['id_count_sampled'] = sampled_id_ct
    res['event_count'] = sampled_event_ct
    # res['event_count_sampled'] = sampled_event_ct
    res['eval_cases'] = sampled_eval_cases

    save_json(res, f'{strategy}.json')

    return eval_cases




def prepare_multiple_household(strategy, clothes, camera, household_dict, query):

    print(f'\nProcessing case: {strategy}...')
    eval_cases = {} # key is identity_id, value is a list of test cases for this identity_id
    total_eval_ct = 0
    id_ct = defaultdict(int)
    event_ct = defaultdict(int)
    for q in tqdm(query.keys()):
        identity_id = parse_id(q)
        household_id = parse_household_id(identity_id)
        mac_addr = parse_mac_addr(q)
        event_id = parse_event_id(q)

        if camera == 'same': # same camera mac address
            if mac_addr not in household_dict[household_id][identity_id]:
                continue
            positive_pool = household_dict[household_id][identity_id][mac_addr]

            # negatives are from other members in the same household with the same camera mac address
            negative_pool = []
            for identity in household_dict[household_id]:
                if identity != identity_id:
                    if mac_addr in household_dict[household_id][identity]:
                        negative_pool += household_dict[household_id][identity][mac_addr]

        else: # cross camera, different camera mac address
            positive_pool = []
            for mac in household_dict[household_id][identity_id]:
                if mac != mac_addr:
                    positive_pool += household_dict[household_id][identity_id][mac]

            # negatives are from other members in the same household with different camera mac address
            negative_pool = []
            for identity in household_dict[household_id]:
                if identity != identity_id:
                    for mac in household_dict[household_id][identity]:
                        if mac != mac_addr:
                            negative_pool += household_dict[household_id][identity][mac]

        # randomly sample 1 positive example from positive pool
        # and 4 negatives from the negative pool
        g = []
        if len(positive_pool) > 1 and len(negative_pool) > 4:
            id_ct[identity_id] += 1
            event_ct[event_id] += 1
            positive_example = random.choice(positive_pool)
            negative_examples = random.sample(negative_pool, 4)
            g.append((positive_example, 1))
            g += [(neg, 0) for neg in negative_examples]

            # shuffle the gallery examples
            random.shuffle(g)
            # get the label, which is the index of the positive example in the gallery
            label  = g.index((positive_example, 1)) + 1
            eval_cases.append({
                'query': q,
                'gallery': [x[0] for x in g],
                'label': label
            })

    print(f"Multi-household: {len(eval_cases)} test cases for clothes={clothes} and camera={camera}.")
    print(f"Identity ct: {len(id_ct)}")
    print(f"Event ct: {len(event_ct)}")

    # sort id_ct and event_ct by value
    id_ct = dict(sorted(id_ct.items(), key=lambda item: item[1], reverse=True))
    event_ct = dict(sorted(event_ct.items(), key=lambda item: item[1], reverse=True))

    res = dict()
    res['id_count'] = id_ct
    res['event_count'] = event_ct
    res['eval_cases'] = eval_cases
    save_json(res, f'{case}.json')

    return eval_cases




if __name__ == "__main__":

    json_file = '/home/tian.liu/tian_data/wyze_person_v2_cross_clothes_full_frame/cross_clothes.json'
    with open(json_file, 'r') as f:
        data = json.load(f)
    gallery = data['gallery']
    query = data['queries']

    random.seed(42)

    # extract the household_id from the gallery
    household_dict = get_household_info(gallery)
    save_json(household_dict, 'household_info_v2_cross_clothes.json')

    ## check how many identity_id have more than 1 mac address
    ## all the identity_id should have only 1 unique mac address
    # identity_id_mac_count = defaultdict(set)
    # for household_id in household_dict:
    #     for identity_id in household_dict[household_id]:
    #         for mac_addr in household_dict[household_id][identity_id]:
    #             identity_id_mac_count[identity_id].add(mac_addr)

    # print(f"Total unique identity IDs: {len(identity_id_mac_count)}")
    # print("Identity IDs with more than 1 unique MAC address:")
    # for identity_id, mac_addrs in identity_id_mac_count.items():
    #     if len(mac_addrs) > 1:
    #         print(f"{identity_id}: {len(mac_addrs)} unique MAC addresses")
    # exit()

    strategy = 'singlehousehold_crossclothes_samecamera'
    single_crossclothes_samecamera = prepare_household_gallery(strategy=strategy,
                                                               household_type='single',
                                                            clothes='cross',
                                                            camera='same',
                                                            household_dict=household_dict,
                                                            query=query)


    strategy = 'singlehousehold_crossclothes_crosscamera'
    single_crossclothes_crosscamera = prepare_household_gallery(strategy=strategy,
                                                                household_type='single',
                                                            clothes='cross',
                                                            camera='cross',
                                                            household_dict=household_dict,
                                                            query=query)

    strategy = 'multihousehold_crossclothes_samecamera'
    multihousehold_crossclothes_samecamera = prepare_household_gallery(strategy=strategy,
                                                                       household_type='multiple',
                                                                        clothes='cross',
                                                                        camera='same',
                                                                        household_dict=household_dict,
                                                                        query=query)

    strategy = 'multihousehold_crossclothes_crosscamera'
    multihousehold_crossclothes_crosscamera = prepare_household_gallery(strategy=strategy,
                                                                        household_type='multiple',
                                                                        clothes='cross',
                                                                        camera='cross',
                                                                        household_dict=household_dict,
                                                                        query=query)