//generate mock data
export type MockData = {
    id: number;
    name: string;
    image: string;
    price: number;
    description: string;
};

export type MockNFTData = MockData & {
    tokenId: string;
    tokenContract: string;
};
export const generateMockData = (): MockData[] => {
    return [
        // {
        // 	id: 1,
        // 	name: 'Hard Fork t-shirt',
        // 	image: 'https://img.joomcdn.net/d2613148b39ca93f42398e4d00ba21899bb0386c_original.jpeg',
        // 	price: 0.05,
        // 	description: '',
        // },
        // {
        // 	id: 2,
        // 	name: 'Bankless t-shirt',
        // 	image:
        // 		'https://cdn.shopify.com/s/files/1/0258/8924/3182/products/BanklessT-ShirtFront_1800x1800.png.jpg?v=1646438936',
        // 	price: 0.2,
        // 	description: '',
        // },
        {
            id: 3,
            name: 'EthGlobal tee',
            image: 'https://pbs.twimg.com/media/FHoOQ7BXMAsO7Jb?format=jpg&name=medium',
            price: 0.08,
            description: '',
        },
        {
            id: 4,
            name: 'Bitconnect tee',
            image:
                'https://m.media-amazon.com/images/I/A13usaonutL._CLa%7C2140%2C2000%7C612TwNUg7DL.png%7C0%2C0%2C2140%2C2000%2B0.0%2C0.0%2C2140.0%2C2000.0_AC_UX679_.png',
            price: 0.1,
            description: '',
        },
        // {
        // 	id: 5,
        // 	name: '3AC sweatshirt',
        // 	image:
        // 		'https://teeruto.com/wp-content/uploads/2022/08/lehman-brothers-three-arrows-capital-2022-risk-management-department-unisex-sweatshirtizme0.jpg',
        // 	price: 5,
        // 	description: '',
        // },
        {
            id: 6,
            name: 'Llama tee',
            image: 'https://cdn.shopify.com/s/files/1/0844/9673/products/Cat_and_Llama_TS_1000x.jpg?v=1573986430',
            price: 0.1,
            description: '',
        },
    ];
};
